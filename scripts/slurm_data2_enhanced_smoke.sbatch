#!/bin/bash
#SBATCH --job-name=stonefly-d2-smoke
#SBATCH --account=mfk-gsr2
#SBATCH --partition=ISP
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p logs/slurm backend/saved_models_data2_enhanced_smoke

if [ ! -f "data/final_stonefly_dataset.csv" ]; then
  echo "ERROR: data/final_stonefly_dataset.csv 不存在。"
  echo "该数据文件被 .gitignore 忽略，git 同步后需要单独上传到集群。"
  exit 2
fi

if [ ! -f "data2/occurrence.txt" ]; then
  echo "ERROR: data2/occurrence.txt 不存在。"
  echo "data2 数据包被 .gitignore 忽略，git 同步后需要单独上传 data2 目录到集群。"
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# data2 增强快速试跑：
# - 默认只训练 XGBoost 和 RandomForest
# - 默认抽样 3000 条训练样本
# - 主要用于确认环境、数据路径和训练流程是否正常
PYTHONUNBUFFERED=1 python -u backend/train_data2_enhanced.py \
  --main-data data/final_stonefly_dataset.csv \
  --data2-occurrence data2/occurrence.txt \
  --output-dir backend/saved_models_data2_enhanced_smoke \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-3000}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --models "${MODELS:-xgboost,random_forest}"

