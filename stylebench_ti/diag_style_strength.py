"""诊断：TI 风格 token 对生成结果的实际影响力。

同内容、同 seed、同采样参数，对比三种 prompt：
  1. base            —— 不加风格 token
  2. tail            —— 当前做法："{content} in the style of <token>"
  3. front_repeat    —— "<token> style. <token> style. {content}"（加权风格）
输出到 Textual Inversion-result/_diag/，用于判断 token 是否有实质作用。
"""

import json
import os

import torch
from PIL import Image

import run_ti as R
import ti_config as C

USER = "U0031"
CONTENT = "c100118"
OUT_DIR = os.path.join(C.OUTPUT_ROOT, "_diag")


def main() -> None:
    plan = [json.loads(l) for l in open(C.PLAN_FILE, encoding="utf-8") if l.strip()]
    item = next(p for p in plan if p["user_id"] == USER and p["content_id"] == CONTENT)
    content_prompt = item["prompt_en"].rstrip(".")
    os.makedirs(OUT_DIR, exist_ok=True)

    pipe = R.load_pipe()
    token = R.apply_embedding(pipe, USER)

    variants = {
        "1_base": content_prompt + ".",
        "2_tail": f"{content_prompt} in the style of {token}.",
        "3_front_repeat": f"{token} style. {token} style. {content_prompt}.",
    }

    for name, prompt in variants.items():
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
        path = os.path.join(OUT_DIR, f"{USER}_{CONTENT}_{name}.png")
        image.save(path)
        print(f"[{name}] -> {path}\n    prompt: {prompt}", flush=True)


if __name__ == "__main__":
    main()
