"""Textual Inversion + SDXL：按用户训练风格 embedding，并生成测试集图像。

每个用户：
  1. 用 ti_data/<user_id>/ 下最多 5 张 stage1 4–5 分图像训练一个占位词 embedding；
  2. 用该 embedding + 内容 prompt 生成测试集对应内容的图像。

支持分片并行与断点续跑（已存在的 embedding / 图片会跳过）。
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from safetensors.torch import load_file

import ti_config as C


def train_user(user_id: str, force: bool = False) -> str:
    out_dir = os.path.join(C.EMBED_DIR, user_id)
    emb_path = os.path.join(out_dir, "learned_embeds.safetensors")
    if os.path.exists(emb_path) and not force:
        return emb_path
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        C.TRAINER_SCRIPT,
        "--pretrained_model_name_or_path", C.SDXL_DIFFUSERS,
        "--train_data_dir", os.path.join(C.DATA_DIR, user_id),
        "--output_dir", out_dir,
        "--placeholder_token", C.placeholder_token(user_id),
        "--initializer_token", C.INITIALIZER_TOKEN,
        "--learnable_property", C.LEARNABLE_PROPERTY,
        "--resolution", str(C.TRAIN_RESOLUTION),
        "--train_batch_size", str(C.TRAIN_BATCH_SIZE),
        "--max_train_steps", str(C.TRAIN_STEPS),
        "--learning_rate", str(C.TRAIN_LR),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--center_crop",
        "--repeats", "100",
        "--seed", str(C.TRAIN_SEED),
        "--checkpointing_steps", "500",
    ]
    subprocess.run(cmd, check=True)
    return emb_path


def load_pipe(device: str = "cuda") -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_single_file(
        C.SDXL_CKPT,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_tiling()
    return pipe.to(device)


def apply_embedding(pipe, user_id: str) -> str:
    """把用户学到的占位向量同时写入 SDXL 的两个文本编码器。

    SDXL 的文本条件 = concat(enc1 768 维, enc2 1280 维)，且全局 pooled 向量只来自 enc2；
    只注入 enc1 会让不同用户的全局风格几乎相同，因此两个编码器都必须注入。
    """
    token = C.placeholder_token(user_id)
    state = load_file(os.path.join(C.EMBED_DIR, user_id, "learned_embeds.safetensors"))
    if token not in state or f"{token}__enc2" not in state:
        raise RuntimeError(
            f"{user_id}: embedding 缺少 enc1/enc2 向量（只有 enc1 说明是旧版 trainer 训练的），请重新训练"
        )
    emb1 = state[token].reshape(-1)
    emb2 = state[f"{token}__enc2"].reshape(-1)

    # ---- encoder 1 ----
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == pipe.tokenizer.unk_token_id:
        pipe.tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    weight = pipe.text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        weight.data[token_id] = emb1.to(weight.device, dtype=weight.dtype)

    # ---- encoder 2 ----
    token_id_2 = pipe.tokenizer_2.convert_tokens_to_ids(token)
    if token_id_2 is None or token_id_2 == pipe.tokenizer_2.unk_token_id:
        pipe.tokenizer_2.add_tokens([token])
        pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
        token_id_2 = pipe.tokenizer_2.convert_tokens_to_ids(token)
    weight_2 = pipe.text_encoder_2.get_input_embeddings().weight
    with torch.no_grad():
        weight_2.data[token_id_2] = emb2.to(weight_2.device, dtype=weight_2.dtype)

    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=C.PLAN_FILE, help="生成计划 jsonl 路径")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-gen", action="store_true")
    parser.add_argument("--data-dir", default=C.DATA_DIR, help="每用户训练图目录（可指向 3 图目录）")
    parser.add_argument("--emb-dir", default=C.EMBED_DIR, help="embedding 输出目录")
    parser.add_argument("--out-root", default=C.OUTPUT_ROOT, help="生成图根目录")
    parser.add_argument("--results-jsonl", default=C.RESULTS_JSONL, help="结果记录 jsonl")
    parser.add_argument("--max-refs", type=int, default=None, help="仅写入元数据：实际使用的参考图数量")
    parser.add_argument("--select-mode", default="rating45_hash", help="仅写入元数据：参考图选择规则")
    args = parser.parse_args()

    C.DATA_DIR = args.data_dir
    C.EMBED_DIR = args.emb_dir
    C.OUTPUT_ROOT = args.out_root
    C.RESULTS_JSONL = args.results_jsonl

    plan = [json.loads(line) for line in open(args.plan, encoding="utf-8") if line.strip()]
    plan = [item for i, item in enumerate(plan) if i % args.num_shards == args.shard]
    if args.limit is not None:
        plan = plan[: args.limit]

    os.makedirs(C.OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(C.RESULTS_JSONL), exist_ok=True)
    proto = C.protocol_dict(max_refs=args.max_refs, select_mode=args.select_mode)

    # 阶段 A：先训练全部用户的 embedding。
    # 此阶段父进程不加载生成 pipeline，否则会与训练子进程争抢同一张卡的显存导致 OOM。
    if not args.skip_train:
        for idx, item in enumerate(plan, 1):
            uid = item["user_id"]
            t0 = time.time()
            emb = train_user(uid, force=args.force_train)
            print(f"[train {idx}/{len(plan)}] {uid} ({time.time()-t0:.1f}s) -> {emb}", flush=True)
        torch.cuda.empty_cache()

    if args.skip_gen:
        print(f"分片 {args.shard}/{args.num_shards} 仅训练完成，共 {len(plan)} 个样本", flush=True)
        return

    # 阶段 B：训练全部结束后，只加载一次 pipeline，统一生成。
    def out_path_of(item) -> str:
        # 计划里带 group 时按组归档：<out_root>/<group>/<user>/<content>.png
        group = item.get("group") or ""
        d = os.path.join(C.OUTPUT_ROOT, group, item["user_id"]) if group else os.path.join(C.OUTPUT_ROOT, item["user_id"])
        return os.path.join(d, f"{item['content_id']}.png")

    todo = []
    for item in plan:
        out_path = out_path_of(item)
        if os.path.exists(out_path) and not args.force_gen:
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        todo.append(item)

    if not todo:
        print(f"分片 {args.shard}/{args.num_shards} 无需生成，共 {len(plan)} 个样本", flush=True)
        return

    pipe = load_pipe()
    for idx, item in enumerate(todo, 1):
        uid, cid = item["user_id"], item["content_id"]
        out_path = out_path_of(item)
        token = apply_embedding(pipe, uid)
        prompt = f"{item['prompt_en'].rstrip('.')} " + C.PROMPT_SUFFIX.format(token=token)

        generator = torch.Generator(device="cuda").manual_seed(C.GEN_SEED)
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            negative_prompt=C.NEGATIVE_PROMPT,
            num_inference_steps=C.NUM_STEPS,
            guidance_scale=C.CFG,
            width=C.WIDTH,
            height=C.HEIGHT,
            generator=generator,
        ).images[0]

        meta = PngInfo()
        meta.add_text("prompt", prompt)
        meta.add_text("seed", str(C.GEN_SEED))
        meta.add_text("algorithm", "textual_inversion")
        meta.add_text("user_id", uid)
        meta.add_text("content_id", cid)
        meta.add_text("placeholder_token", token)
        image.save(out_path, pnginfo=meta)

        record = {
            "algorithm": "textual_inversion",
            "user_id": uid,
            "content_id": cid,
            "prompt": prompt,
            "seed": C.GEN_SEED,
            "placeholder_token": token,
            "embedding": os.path.join(C.EMBED_DIR, uid, "learned_embeds.safetensors"),
            "output": out_path,
            "elapsed_sec": round(time.time() - t0, 2),
            "protocol": proto,
        }
        with open(C.RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[gen {idx}/{len(todo)}] {uid}/{cid} -> {out_path}", flush=True)

    print(f"分片 {args.shard}/{args.num_shards} 处理完成，共 {len(plan)} 个样本", flush=True)


if __name__ == "__main__":
    main()
