#!/bin/bash
#SBATCH --job-name=stonefly-ds-opt
#SBATCH --account=mfk-gsr2
#SBATCH --partition=ISP
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p logs/slurm backend/saved_models/dataset_accuracy_optimizer

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA_ARGS=()
if [ "${QUICK:-0}" = "1" ]; then
  # QUICK=1 时跳过较慢的 SVM，只用 RandomForest、XGBoost、KNN 快速筛选方向。
  EXTRA_ARGS+=(--quick)
fi

if [ "${SAVE_BEST_DATASET:-1}" = "1" ]; then
  # 保存当前最佳候选数据集，便于后续针对该范围做正式训练或给客户说明范围。
  EXTRA_ARGS+=(--save-best-dataset)
fi

PYTHONUNBUFFERED=1 python -u backend/experiments/dataset_accuracy_optimizer.py \
  --main-data data/final_stonefly_dataset.csv \
  --data2-occurrence data2/occurrence.txt \
  --output-dir backend/saved_models/dataset_accuracy_optimizer \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-20000}" \
  "${EXTRA_ARGS[@]}"

