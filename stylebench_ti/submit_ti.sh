#!/bin/bash
# 提交一个 Textual Inversion 分片：先训练若干用户的 embedding，再生成各自图像
# 用法：sbatch submit_ti.sh <shard_id> <num_shards>
#SBATCH --job-name=stylebench_ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=GPU1
#SBATCH --time=06:00:00
#SBATCH --output=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/ti_%j.out
#SBATCH --error=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/ti_%j.err

set -euo pipefail

SHARD=${1:?用法: sbatch submit_ti.sh <shard_id> <num_shards>}
NUM=${2:-4}

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data-nfs/gpu4-0/ui202420199/.conda/envs/MZ

cd /data-nfs/gpu4-0/ui202420199/stylebench_ti
export HF_HOME=/data-nfs/gpu4-0/ui202420199/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "node=$(hostname) shard=${SHARD}/${NUM} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python run_ti.py --shard "${SHARD}" --num-shards "${NUM}"
