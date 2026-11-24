"""将 SDXL 单文件权重转换为 diffusers 目录格式（Textual Inversion 训练脚本需要）。"""

import os

import torch
from diffusers import StableDiffusionXLPipeline

import ti_config as C


def main() -> None:
    if os.path.exists(os.path.join(C.SDXL_DIFFUSERS, "model_index.json")):
        print(f"已存在，跳过：{C.SDXL_DIFFUSERS}")
        return
    print(f"从 {C.SDXL_CKPT} 转换 ...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        C.SDXL_CKPT,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    )
    pipe.save_pretrained(C.SDXL_DIFFUSERS, safe_serialization=True)
    print(f"完成：{C.SDXL_DIFFUSERS}")


if __name__ == "__main__":
    main()
