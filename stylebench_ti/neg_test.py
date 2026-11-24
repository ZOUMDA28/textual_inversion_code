"""负向提示词 A/B 对照：同一用户、同一 seed，只改 negative prompt。

用途：检验输出普遍偏灰/低饱和是否由负向词里的 "saturation, low contrast" 造成。
"""

import argparse
import os

import torch
from PIL import PngImagePlugin

import ti_config as C
import run_ti as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--content", default="c100118")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--emb-dir", default=C.SMOKE_EMBED_DIR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    C.EMBED_DIR = args.emb_dir
    variants = {
        "A_baseline": C.NEGATIVE_PROMPT,
        "B_no_sat": "text, watermark, lowres, low quality, worst quality, deformed, glitch, noisy, blurry",
        "C_none": "",
    }

    pipe = R.load_pipe()
    token = R.apply_embedding(pipe, args.user)
    prompt = f"{args.prompt.rstrip('.')} " + C.PROMPT_SUFFIX.format(token=token)
    os.makedirs(args.out, exist_ok=True)

    for name, neg in variants.items():
        g = torch.Generator(device="cuda").manual_seed(C.GEN_SEED)
        img = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=C.NUM_STEPS,
            guidance_scale=C.CFG,
            width=C.WIDTH,
            height=C.HEIGHT,
            generator=g,
        ).images[0]
        path = os.path.join(args.out, f"{args.user}_{args.content}_{name}.png")
        meta = PngImagePlugin.PngInfo()
        meta.add_text("prompt", prompt)
        meta.add_text("negative_prompt", neg)
        meta.add_text("seed", str(C.GEN_SEED))
        img.save(path, pnginfo=meta)
        print(f"[{name}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
