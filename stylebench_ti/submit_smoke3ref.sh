#!/bin/bash
# 3 参考图 smoke test：12 用户（G1 随机6 / G2 G14抽3 / G3 G19抽3），同一内容 c100118
# 用法: sbatch submit_smoke3ref.sh <shard 0..3>
#SBATCH --job-name=ti_smoke3
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=GPU1
#SBATCH --time=03:00:00
#SBATCH --output=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/smoke3_%j.out
#SBATCH --error=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/smoke3_%j.err
set -euo pipefail
SHARD=${1:?用法: sbatch submit_smoke3ref.sh <shard 0..3>}
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data-nfs/gpu4-0/ui202420199/.conda/envs/MZ
cd /data-nfs/gpu4-0/ui202420199/stylebench_ti
export HF_HOME=/data-nfs/gpu4-0/ui202420199/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
echo "node=$(hostname) shard=${SHARD}"
python run_ti.py \
  --plan smoke3ref_plan.jsonl \
  --shard "${SHARD}" --num-shards 4 \
  --data-dir ti_data_3ref \
  --emb-dir ti_embeddings_3ref \
  --out-root "/data-nfs/gpu4-0/ui202420199/Textual Inversion-result/_smoke3ref" \
  --results-jsonl "/data-nfs/gpu4-0/ui202420199/Textual Inversion-result/_smoke3ref/results.jsonl" \
  --max-refs 3 --select-mode rating5_reverse
