#!/bin/bash
# ============================================================
# SLURM 正式训练脚本 — 增强数据集 4 模型
# 用途：在集群上使用 stonefly_combined_data_augmented.csv 训练
#       XGBoost(目标90%) + RF(目标82%) + KNN + SVM
#
# 提交方式：
#   sbatch scripts/slurm_train_augmented.sbatch
#
# 可选覆盖参数（提交时用 --export 传入）：
#   SKIP_SVM=1        跳过 SVM，大幅缩短时间
#   SKIP_KNN=1        跳过 KNN
#   SKIP_HIER=1       跳过 family -> species 层级模型
#   XGB_TREES=1200    XGBoost 树的棵数（默认 1500）
#   XGB_DEPTH=10      XGBoost 最大深度（默认 10）
#   RF_TREES=600      RF 树的棵数（默认 600）
#   RF_CAP=80000      RF 最大训练样本（默认 80000，过大会 OOM）
#   RF_MAX_DEPTH=30   RF 最大深度（默认 30，None=不限会 OOM）
#   RF_JOBS=8         RF 并行数（默认 8，独立于全局 n_jobs）
#   SVM_CAP=30000     SVM 最大训练样本（默认 30000）
#   KNN_CAP=50000     KNN 最大训练样本（默认 50000）
#   HIER_CAP=80000    层级模型最大训练样本数（默认 80000）
#   HIER_JOBS=4       层级模型并行线程数（默认 4）
#
# ⚠️  RF 内存注意：max_depth=None + 大样本 + 多并行 → 容易 OOM
#     如果仍然 OOM，调低 RF_CAP=50000 或 RF_MAX_DEPTH=20
#
# 示例（跳过 SVM 加速实验）：
#   sbatch --export=ALL,SKIP_SVM=1 scripts/slurm_train_augmented.sbatch
# ============================================================

#SBATCH --job-name=stonefly-augmented
#SBATCH --account=mfk-gsr2
#SBATCH --partition=plcyf-com
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=0-12:00:00
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
echo "内存      : 80G"
echo "开始时间  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 数据文件检查 ──────────────────────────────────────────────────────────────
DATA_FILE="data/stonefly_combined_data_augmented.csv"
if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: $DATA_FILE 不存在。"
  echo "  该文件已在 git 中追踪，请确认 git pull 后文件存在。"
  echo "  若文件过大未能同步，需手动上传至集群。"
  exit 2
fi
echo "数据文件已找到: $DATA_FILE  ($(du -sh $DATA_FILE | cut -f1))"

# ── Python 环境 ───────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  echo "创建虚拟环境..."
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
echo "Python: $($VENV_PYTHON --version)  路径: $VENV_PYTHON"

echo "安装/更新依赖..."
"$VENV_PIP" install --upgrade pip -q
"$VENV_PIP" install -r backend/requirements.txt -q

# ── 线程数绑定（防止 BLAS 过度抢占）─────────────────────────────────────────
NCPU="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"
echo "线程数绑定: OMP/MKL/OPENBLAS = $NCPU"

# ── 构造训练命令 ──────────────────────────────────────────────────────────────
cd backend

CMD="$VENV_PYTHON -u train_augmented.py \
  --data-path   ../data/stonefly_combined_data_augmented.csv \
  --output-dir  saved_models_augmented \
  --viz-dir     visualizations_augmented \
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
  --hierarchical-cap ${HIER_CAP:-80000} \
  --hierarchical-jobs ${HIER_JOBS:-4}"

# 可选跳过慢模型
[ "${SKIP_SVM:-0}" = "1" ] && CMD="$CMD --skip-svm"
[ "${SKIP_KNN:-0}" = "1" ] && CMD="$CMD --skip-knn"
[ "${SKIP_HIER:-0}" = "1" ] && CMD="$CMD --skip-hierarchical"

echo ""
echo "执行命令: $CMD"
echo "========================================================"
PYTHONUNBUFFERED=1 eval $CMD

echo ""
echo "========================================================"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "输出目录: backend/saved_models_augmented/"
echo ""
echo "评估结果摘要:"
if [ -f "saved_models_augmented/model_evaluation_report.json" ]; then
  "$VENV_PYTHON" -c "
import json, sys
r = json.load(open('saved_models_augmented/model_evaluation_report.json'))
ev = r['evaluation_results']
print(f\"{'模型':<16} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'F1-macro':>9}\")
print('-' * 54)
for name, res in ev.items():
    t3 = f\"{res['top3_accuracy']:.4f}\" if res.get('top3_accuracy') else '  N/A'
    t5 = f\"{res['top5_accuracy']:.4f}\" if res.get('top5_accuracy') else '  N/A'
    print(f\"{name:<16} {res['top1_accuracy']:>7.4f} {t3:>7} {t5:>7} {res['f1_macro']:>9.4f}\")
"
fi
if [ -f "saved_models_augmented/hierarchical_results.json" ]; then
  echo ""
  echo "层级模型结果摘要:"
  "$VENV_PYTHON" -c "
import json
r = json.load(open('saved_models_augmented/hierarchical_results.json'))
m = r['metrics']
print(f\"Family Top-1: {m['family_accuracy']:.4f}\")
print(f\"Species Top-1: {m['hierarchical_species_accuracy']:.4f}\")
print(f\"Species Top-4: {m['hierarchical_species_top_4_accuracy']:.4f}\")
print(f\"Oracle Top-1 : {m['oracle_family_species_accuracy']:.4f}\")
"
fi
echo "========================================================"
