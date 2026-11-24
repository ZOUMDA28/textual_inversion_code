"""诊断 2：换用加权 prompt 模板，检验能否拉开用户间风格差异。

对同一内容 c100118，用三个已有 embedding，各生成一张：
  prompt = "<token> style. <token> style. {content}"
输出到 Textual Inversion-result/_diag2/
"""

import json

import torch

import run_ti as R
import ti_config as C

USERS = ["U0031", "U0185", "U0330"]
CONTENT = "c100118"


def main() -> None:
    import os

    plan = [json.loads(l) for l in open(C.PLAN_FILE, encoding="utf-8") if l.strip()]
    out_dir = os.path.join(C.OUTPUT_ROOT, "_diag2")
    os.makedirs(out_dir, exist_ok=True)

    pipe = R.load_pipe()
    for uid in USERS:
        item = next(p for p in plan if p["user_id"] == uid and p["content_id"] == CONTENT)
        content = item["prompt_en"].rstrip(".")
        token = R.apply_embedding(pipe, uid)
        prompt = f"{token} style. {token} style. {content}."
        generator = torch.Generator(device="cuda").manual_seed(C.GEN_SEED)
        image = pipe(
            prompt=prompt,
            negative_prompt=C.NEGATIVE_PROMPT,
            num_inference_steps=C.NUM_STEPS,
            guidance_scale=C.CFG,
            width=C.WIDTH,
            height=C.HEIGHT,
            generator=generator,
        ).images[0]
        path = os.path.join(out_dir, f"{uid}_{CONTENT}_front2.png")
        image.save(path)
        print(f"[{uid}] -> {path}\n    prompt: {prompt}", flush=True)


if __name__ == "__main__":
    main()
