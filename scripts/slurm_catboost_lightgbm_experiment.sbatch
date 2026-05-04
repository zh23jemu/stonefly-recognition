#!/bin/bash
#SBATCH --job-name=stonefly-exp
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p logs/slurm backend/saved_models/model_experiments

if [ ! -f "data/final_stonefly_dataset.csv" ]; then
  echo "ERROR: data/final_stonefly_dataset.csv 不存在。"
  echo "该数据文件被 .gitignore 忽略，git 同步后需要单独上传到集群。"
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-experiments.txt

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python backend/experiments/catboost_lightgbm_ensemble_sweep.py \
  --resume \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-20000}" \
  --progress-dir backend/saved_models/model_experiments \
  --state-file backend/saved_models/model_experiments/catboost_lightgbm_ensemble_state.json \
  --result-file backend/saved_models/model_experiments/catboost_lightgbm_ensemble_results.json
