#!/bin/bash
# 同一内容 c100118、三个用户并行训练+生成（验证用户风格是否区分）
#SBATCH --job-name=ti_verify3
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=GPU1
#SBATCH --time=02:00:00
#SBATCH --output=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/v3_%j.out
#SBATCH --error=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/v3_%j.err
set -euo pipefail
SHARD=${1:?用法: sbatch submit_verify3.sh <shard 0..2>}
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data-nfs/gpu4-0/ui202420199/.conda/envs/MZ
cd /data-nfs/gpu4-0/ui202420199/stylebench_ti
export HF_HOME=/data-nfs/gpu4-0/ui202420199/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
echo "node=$(hostname) shard=${SHARD}"
python run_ti.py --plan verify_plan_c100118.jsonl --shard "${SHARD}" --num-shards 3 --force-train --force-gen
