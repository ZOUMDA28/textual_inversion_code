#!/bin/bash
# 单用户 smoke 测试：只跑计划中的第 1 个样本（训练 400/2000 步 + 生成）
# 用法：sbatch submit_smoke.sh
#SBATCH --job-name=ti_smoke
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=GPU1
#SBATCH --time=02:00:00
#SBATCH --output=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/smoke_%j.out
#SBATCH --error=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/smoke_%j.err

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data-nfs/gpu4-0/ui202420199/.conda/envs/MZ

cd /data-nfs/gpu4-0/ui202420199/stylebench_ti
export HF_HOME=/data-nfs/gpu4-0/ui202420199/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python run_ti.py --shard 0 --num-shards 1 --limit 3 --force-train --force-gen
