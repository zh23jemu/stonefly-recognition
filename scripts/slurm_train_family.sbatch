#!/bin/bash
# ============================================================
# SLURM 科级预测训练脚本 — 4 模型 × 2 数据集
# 用途：训练以 family 为目标的分类模型，对比全量 vs 过滤 unknown 两种数据集
#
# 提交方式：
#   sbatch scripts/slurm_train_family.sbatch
#
# 可选覆盖参数（提交时用 --export 传入）：
#   SKIP_SVM=1        跳过 SVM（节省时间，RBF 核 30k 样本约 30-60 分钟）
#   SKIP_KNN=1        跳过 KNN
#   XGB_TREES=300     XGBoost 树数（默认 300，16 类内存消耗低）
#   XGB_DEPTH=10      XGBoost 最大深度（默认 10）
#   RF_TREES=600      RF 树数（默认 600）
#   RF_CAP=80000      RF 最大训练样本（默认 80000）
#   RF_MAX_DEPTH=30   RF 最大深度（默认 30）
#   RF_JOBS=8         RF 并行数（默认 8）
#   SVM_CAP=30000     SVM 最大训练样本（默认 30000）
#   KNN_CAP=50000     KNN 最大训练样本（默认 50000）
#
# 输出目录：
#   backend/saved_models_family/        — 全量数据（含 unknown）
#   backend/saved_models_family_clean/  — 过滤 unknown 后（推荐上线）
#
# 示例（跳过 SVM）：
#   sbatch --export=ALL,SKIP_SVM=1 scripts/slurm_train_family.sbatch
# ============================================================

#SBATCH --job-name=stonefly-family
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/slurm

echo "========================================================"
echo "作业 ID   : ${SLURM_JOB_ID:-local}"
echo "节点      : ${SLURMD_NODENAME:-localhost}"
echo "CPU 核数  : ${SLURM_CPUS_PER_TASK:-32}"
echo "内存      : 60G"
echo "开始时间  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "目标      : family（16 科分类）"
echo "========================================================"

# ── 数据文件检查 ──────────────────────────────────────────────────────────────
DATA_FILE="data/stonefly_combined_data_augmented.csv"
if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: $DATA_FILE 不存在，请先 git pull 或手动上传。"
  exit 2
fi
echo "数据文件: $DATA_FILE  ($(du -sh $DATA_FILE | cut -f1))"

# ── Python 环境 ───────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  echo "创建虚拟环境..."
  "$PYTHON_BIN" -m venv .venv
fi
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
echo "Python: $($VENV_PYTHON --version)"
"$VENV_PIP" install --upgrade pip -q
"$VENV_PIP" install -r backend/requirements.txt -q

# ── 线程数绑定 ────────────────────────────────────────────────────────────────
NCPU="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"
echo "线程数: $NCPU"

cd backend

# ── 公共参数 ──────────────────────────────────────────────────────────────────
BASE_ARGS="--data-path ../data/stonefly_combined_data_augmented.csv \
  --target      family \
  --n-jobs      $NCPU \
  --seed        42 \
  --xgb-trees   ${XGB_TREES:-300} \
  --xgb-depth   ${XGB_DEPTH:-10} \
  --rf-trees    ${RF_TREES:-600} \
  --rf-cap      ${RF_CAP:-80000} \
  --rf-max-depth ${RF_MAX_DEPTH:-30} \
  --rf-jobs     ${RF_JOBS:-8} \
  --svm-cap     ${SVM_CAP:-30000} \
  --knn-cap     ${KNN_CAP:-50000} \
  --skip-hierarchical"

SKIP_FLAGS=""
[ "${SKIP_SVM:-0}" = "1" ] && SKIP_FLAGS="$SKIP_FLAGS --skip-svm"
[ "${SKIP_KNN:-0}" = "1" ] && SKIP_FLAGS="$SKIP_FLAGS --skip-knn"

# ══════════════════════════════════════════════════════════════
# 任务 A：全量数据集（含 unknown month/habitat，~232k 条）
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "任务 A：全量数据集（含 unknown）→ saved_models_family/"
echo "════════════════════════════════════════════════════════"

PYTHONUNBUFFERED=1 $VENV_PYTHON -u train_augmented.py \
  $BASE_ARGS \
  --output-dir saved_models_family \
  --viz-dir    visualizations_family \
  $SKIP_FLAGS

echo ""
echo "任务 A 完成: $(date '+%Y-%m-%d %H:%M:%S')"
print_report() {
  local dir=$1
  if [ -f "$dir/model_evaluation_report.json" ]; then
    $VENV_PYTHON -c "
import json
r = json.load(open('$dir/model_evaluation_report.json'))
ev = r['evaluation_results']
print(f\"{'模型':<16} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'F1-macro':>9}\")
print('-' * 54)
for name, res in ev.items():
    t3 = f\"{res['top3_accuracy']:.4f}\" if res.get('top3_accuracy') else '  N/A'
    t5 = f\"{res['top5_accuracy']:.4f}\" if res.get('top5_accuracy') else '  N/A'
    print(f\"{name:<16} {res['top1_accuracy']:>7.4f} {t3:>7} {t5:>7} {res['f1_macro']:>9.4f}\")
"
  fi
}
print_report saved_models_family

# ══════════════════════════════════════════════════════════════
# 任务 B：过滤 unknown 数据集（仅合成+data2，~168k 条）
#          month != unknown AND habitat != unknown
#          预期 Top-1 ≈ 90%，推荐用于线上推理
# ══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "任务 B：过滤 unknown → saved_models_family_clean/"
echo "════════════════════════════════════════════════════════"

# 先生成过滤后的临时 CSV，供训练脚本直接读取
CLEAN_DATA="../data/stonefly_combined_data_family_clean.csv"
$VENV_PYTHON -c "
import pandas as pd
df = pd.read_csv('../data/stonefly_combined_data_augmented.csv', low_memory=False)
df_clean = df[(df['month'] != 'unknown') & (df['habitat'] != 'unknown')]
df_clean.to_csv('$CLEAN_DATA', index=False)
print(f'过滤后记录数: {len(df_clean)}  (原始: {len(df)})')
"

PYTHONUNBUFFERED=1 $VENV_PYTHON -u train_augmented.py \
  --data-path   $CLEAN_DATA \
  --target      family \
  --n-jobs      $NCPU \
  --seed        42 \
  --xgb-trees   ${XGB_TREES:-300} \
  --xgb-depth   ${XGB_DEPTH:-10} \
  --rf-trees    ${RF_TREES:-600} \
  --rf-cap      ${RF_CAP:-80000} \
  --rf-max-depth ${RF_MAX_DEPTH:-30} \
  --rf-jobs     ${RF_JOBS:-8} \
  --svm-cap     ${SVM_CAP:-30000} \
  --knn-cap     ${KNN_CAP:-50000} \
  --output-dir  saved_models_family_clean \
  --viz-dir     visualizations_family_clean \
  --skip-hierarchical \
  $SKIP_FLAGS

echo ""
echo "任务 B 完成: $(date '+%Y-%m-%d %H:%M:%S')"
print_report saved_models_family_clean

# ── 汇总对比 ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "结果对比（Top-1）:"
echo "  全量数据  → backend/saved_models_family/"
echo "  过滤unknown → backend/saved_models_family_clean/  ← 推荐上线"
echo "========================================================"
