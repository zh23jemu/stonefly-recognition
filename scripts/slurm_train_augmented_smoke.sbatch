#!/bin/bash
# ============================================================
# 冒烟测试脚本 — 验证训练流程不报错（缩小参数，约 10-20 分钟）
# 只跑 XGBoost + RF，并显式验证增强版层级模型链路
#
# 提交：sbatch scripts/slurm_train_augmented_smoke.sbatch
# 或本地: bash scripts/slurm_train_augmented_smoke.sbatch
# ============================================================

#SBATCH --job-name=stonefly-aug-smoke
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/slurm

echo "========================================================"
echo "[SMOKE TEST] 冒烟测试 — 仅跑 XGBoost + RF（小参数）"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

DATA_FILE="data/stonefly_combined_data_augmented.csv"
if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: $DATA_FILE 不存在，请先上传或 git pull。"
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
"$VENV_PIP" install --upgrade pip -q
"$VENV_PIP" install -r backend/requirements.txt -q

NCPU="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"

cd backend

PYTHONUNBUFFERED=1 "$VENV_PYTHON" -u train_augmented.py \
  --data-path   ../data/stonefly_combined_data_augmented.csv \
  --output-dir  saved_models_augmented_smoke \
  --viz-dir     visualizations_augmented_smoke \
  --n-jobs      "$NCPU" \
  --seed        42 \
  --xgb-trees   200 \
  --xgb-depth   8 \
  --rf-trees    100 \
  --rf-cap      30000 \
  --rf-max-depth 25 \
  --rf-jobs     4 \
  --hierarchical-cap 30000 \
  --hierarchical-jobs 4 \
  --skip-svm \
  --skip-knn

echo ""
echo "冒烟测试完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录: backend/saved_models_augmented_smoke/"
if [ -f "saved_models_augmented_smoke/hierarchical_results.json" ]; then
  echo "层级模型结果摘要:"
  "$VENV_PYTHON" -c "
import json
r = json.load(open('saved_models_augmented_smoke/hierarchical_results.json'))
m = r['metrics']
print(f\"Family Top-1: {m['family_accuracy']:.4f}\")
print(f\"Species Top-1: {m['hierarchical_species_accuracy']:.4f}\")
print(f\"Species Top-4: {m['hierarchical_species_top_4_accuracy']:.4f}\")
"
fi
