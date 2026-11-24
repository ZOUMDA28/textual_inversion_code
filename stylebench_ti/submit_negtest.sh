#!/bin/bash
# 负向提示词 A/B 对照（GPU1）
# 用法: sbatch submit_negtest.sh
#SBATCH --job-name=ti_negtest
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=GPU1
#SBATCH --time=01:00:00
#SBATCH --output=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/neg_%j.out
#SBATCH --error=/data-nfs/gpu4-0/ui202420199/stylebench_ti/logs/neg_%j.err
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data-nfs/gpu4-0/ui202420199/.conda/envs/MZ
cd /data-nfs/gpu4-0/ui202420199/stylebench_ti
export HF_HOME=/data-nfs/gpu4-0/ui202420199/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
OUT="/data-nfs/gpu4-0/ui202420199/Textual Inversion-result/_smoke3ref/_negtest"
P="A female anthropomorphic ferret wearing Jedi robes in a forested city."
echo "node=$(hostname)"
for U in U0070 U0364; do
  python neg_test.py --user "$U" --content c100118 --prompt "$P" --out "$OUT"
done
