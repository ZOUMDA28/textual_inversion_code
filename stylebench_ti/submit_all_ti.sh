#!/bin/bash
# 提交 4 个 Textual Inversion 分片（每片约 15 个用户，串行训练+生成）
# 前置：
#   python convert_sdxl.py     # 一次性：SDXL 单文件 → diffusers 目录
#   python prepare_ti_data.py  # 一次性：选每用户 4–5 分的 5 张训练图
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

for s in 0 1 2 3 4 5 6 7; do
  sbatch submit_ti.sh "${s}" 8
done

squeue -u "$USER"
