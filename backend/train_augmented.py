"""
增强数据集训练脚本（4 基础模型，无层级模型）

特征（11个）：
  lat, lon, country, family, body_length_mm, color, head_feature,
  month, habitat, sex, life_stage

目标：XGBoost Top-1 ≈90%，RF ≈82%，SVM/KNN 较低

样本上限策略（SVM/RBF 无法全量训练）：
  XGBoost : 全量训练集（约 19 万）
  RF       : 150,000
  KNN      : 50,000
  SVM      : 30,000

输出目录：saved_models_augmented/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from ml.preprocessing import DataPreprocessor


# ── 特征列表 ──────────────────────────────────────────────────────────────────
FEATURES = [
    "lat", "lon",
    "country", "family",
    "body_length_mm",
    "color", "head_feature",
    "month", "habitat", "sex", "life_stage",
]
TARGET = "species"
MIN_SPECIES_COUNT = 50   # 过滤极低频物种（增强数据集中实际全部 >=300）


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int,
                  classes: np.ndarray) -> float:
    """计算 Top-K 准确率（真实标签是否出现在前 K 候选中）。"""
    top_k_idx = np.argsort(proba, axis=1)[:, -k:]
    top_k_labels = classes[top_k_idx]
    return float(np.mean([y_true[i] in top_k_labels[i] for i in range(len(y_true))]))


def stratified_subsample(X: np.ndarray, y: np.ndarray,
                         n: int, seed: int = 42):
    """分层抽样，保证每个类别比例相同。"""
    if len(y) <= n:
        return X, y
    spl = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(spl.split(np.zeros(len(y)), y))
    return X[idx], y[idx]


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             name: str, le: LabelEncoder) -> dict:
    """评估单个模型，输出 Top-1/3/5 准确率和 F1。"""
    log(f"评估 {name}...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    top1 = accuracy_score(y_test, y_pred)
    bal  = balanced_accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1m  = f1_score(y_test, y_pred, average="macro",   zero_division=0)

    top3 = top4 = top5 = None
    if hasattr(model, "predict_proba"):
        proba   = model.predict_proba(X_test)
        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        top3 = topk_accuracy(y_test, proba, 3, classes)
        top4 = topk_accuracy(y_test, proba, 4, classes)
        top5 = topk_accuracy(y_test, proba, 5, classes)

    elapsed = time.time() - t0
    result = {
        "top1_accuracy":      round(top1, 4),
        "balanced_accuracy":  round(bal,  4),
        "f1_weighted":        round(f1w,  4),
        "f1_macro":           round(f1m,  4),
        "top3_accuracy":      round(top3, 4) if top3 is not None else None,
        "top4_accuracy":      round(top4, 4) if top4 is not None else None,
        "top5_accuracy":      round(top5, 4) if top5 is not None else None,
        "eval_time_sec":      round(elapsed, 1),
    }

    log(f"  Top-1={top1:.4f}  Bal={bal:.4f}  F1w={f1w:.4f}  F1m={f1m:.4f}")
    if top3:
        log(f"  Top-3={top3:.4f}  Top-4={top4:.4f}  Top-5={top5:.4f}")
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="增强数据集4模型训练脚本")
    p.add_argument("--data-path",   default="../data/stonefly_combined_data_augmented.csv")
    p.add_argument("--output-dir",  default="saved_models_augmented")
    p.add_argument("--viz-dir",     default="visualizations_augmented")
    p.add_argument("--n-jobs",      type=int, default=-1,
                   help="并行核数，-1 表示全部 CPU")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--skip-svm",    action="store_true",
                   help="跳过 SVM（节省时间）")
    p.add_argument("--skip-knn",    action="store_true",
                   help="跳过 KNN（节省时间）")
    p.add_argument("--xgb-trees",   type=int, default=1500,
                   help="XGBoost n_estimators")
    p.add_argument("--xgb-depth",   type=int, default=10,
                   help="XGBoost max_depth")
    p.add_argument("--rf-trees",     type=int, default=600,
                   help="RF n_estimators")
    p.add_argument("--rf-cap",       type=int, default=80000,
                   help="RF 最大训练样本数（减小可降低内存）")
    p.add_argument("--rf-max-depth", type=int, default=30,
                   help="RF max_depth，None 表示不限（不限会 OOM）")
    p.add_argument("--rf-jobs",      type=int, default=8,
                   help="RF 专用并行数，独立于 --n-jobs（RF 内存随并行数线性增长）")
    p.add_argument("--svm-cap",     type=int, default=30000,
                   help="SVM 最大训练样本数")
    p.add_argument("--knn-cap",     type=int, default=50000,
                   help="KNN 最大训练样本数")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.viz_dir,   exist_ok=True)
    seed = args.seed

    log("=" * 60)
    log("石蝇分类系统 — 增强数据集训练（4 基础模型）")
    log("=" * 60)

    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    log(f"加载数据集: {args.data_path}")
    df = pd.read_csv(args.data_path, low_memory=False)
    log(f"  原始行数: {len(df)}  列数: {len(df.columns)}")

    # month 列可能是 int/str 混合，统一转为字符串分类
    df["month"] = df["month"].astype(str)

    # 只保留目标列和特征列
    keep_cols = FEATURES + [TARGET]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 过滤极低频物种
    counts = df[TARGET].value_counts()
    valid  = counts[counts >= MIN_SPECIES_COUNT].index
    df     = df[df[TARGET].isin(valid)].reset_index(drop=True)
    log(f"  过滤后行数: {len(df)}  物种数: {df[TARGET].nunique()}")

    # ── 2. 数据划分 ────────────────────────────────────────────────────────────
    log("分层划分训练/测试/验证集 (70/15/15)...")
    y_all = df[TARGET].to_numpy()

    # 先拆出 30% holdout
    spl1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(spl1.split(np.zeros(len(y_all)), y_all))
    df_train   = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    # holdout 一分为二：test / validation
    y_holdout = df_holdout[TARGET].to_numpy()
    spl2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    test_idx, val_idx = next(spl2.split(np.zeros(len(y_holdout)), y_holdout))
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    df_val  = df_holdout.iloc[val_idx].reset_index(drop=True)

    log(f"  训练集: {len(df_train)}  测试集: {len(df_test)}  验证集: {len(df_val)}")

    # ── 3. 预处理 ──────────────────────────────────────────────────────────────
    log("拟合预处理器（仅在训练集上）...")
    preprocessor = DataPreprocessor()
    X_train_full, y_train, _ = preprocessor.preprocess_pipeline(df_train, TARGET)
    X_test_full,  y_test,  _ = preprocessor.transform_pipeline(df_test,  TARGET)
    X_val_full,   y_val,   _ = preprocessor.transform_pipeline(df_val,   TARGET)

    # DataFrame → numpy（XGBoost / SVM / KNN 都接受 numpy）
    X_train_np = X_train_full.to_numpy(dtype=np.float32)
    X_test_np  = X_test_full.to_numpy(dtype=np.float32)
    X_val_np   = X_val_full.to_numpy(dtype=np.float32)

    preprocessor.save(os.path.join(args.output_dir, "preprocessor.pkl"))
    log(f"  预处理完成，特征维度: {X_train_np.shape[1]}")

    # XGBoost 需要整数标签 0..N-1
    le_target = preprocessor.target_encoder
    np.savez(os.path.join(args.output_dir, "test_data.npz"),
             X_test=X_test_np, y_test=y_test)
    np.savez(os.path.join(args.output_dir, "validation_data.npz"),
             X_validation=X_val_np, y_validation=y_val)

    # 元信息
    meta = {
        "data_path":          args.data_path,
        "n_species":          int(df[TARGET].nunique()),
        "train_samples":      len(X_train_np),
        "test_samples":       len(X_test_np),
        "val_samples":        len(X_val_np),
        "features":           list(X_train_full.columns),
        "n_features":         X_train_np.shape[1],
        "min_species_count":  MIN_SPECIES_COUNT,
        "seed":               seed,
        "timestamp":          datetime.now().isoformat(),
    }
    json.dump(meta, open(os.path.join(args.output_dir, "dataset_split_metadata.json"), "w"),
              indent=2, ensure_ascii=False)

    evaluation_results = {}
    training_log       = {}

    # ── 4. XGBoost（全量，目标 ≈90%）─────────────────────────────────────────
    log("\n" + "=" * 50)
    log(f"训练 XGBoost  样本={len(X_train_np)}  trees={args.xgb_trees}  depth={args.xgb_depth}")

    # XGBoost 需要整数标签
    le_xgb   = LabelEncoder()
    y_tr_xgb = le_xgb.fit_transform(y_train)
    y_te_xgb = le_xgb.transform(y_test)

    sw_xgb = compute_sample_weight("balanced", y_tr_xgb)

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators       = args.xgb_trees,
        max_depth          = args.xgb_depth,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        tree_method        = "hist",       # 高效直方图，支持大数据集
        n_jobs             = args.n_jobs,
        random_state       = seed,
        eval_metric        = "mlogloss",
        verbosity          = 1,
    )
    xgb.fit(X_train_np, y_tr_xgb, sample_weight=sw_xgb)
    xgb_time = time.time() - t0
    log(f"  XGBoost 训练耗时: {xgb_time/60:.1f} 分钟")

    joblib.dump(xgb,    os.path.join(args.output_dir, "xgboost_model.pkl"))
    joblib.dump(le_xgb, os.path.join(args.output_dir, "xgboost_label_encoder.pkl"))

    # 评估（XGBoost classes_ 是整数编码，对应 le_xgb.classes_）
    xgb_res = evaluate(xgb, X_test_np, y_te_xgb, "XGBoost",
                       le=le_xgb)
    evaluation_results["xgboost"] = xgb_res
    training_log["xgboost"] = {"training_time_sec": round(xgb_time, 1),
                                "n_train": len(X_train_np),
                                "params": {"n_estimators": args.xgb_trees,
                                           "max_depth": args.xgb_depth,
                                           "learning_rate": 0.05}}

    # ── 5. Random Forest（限 RF_CAP 样本，目标 ≈82%）──────────────────────────
    log("\n" + "=" * 50)
    X_rf, y_rf = stratified_subsample(X_train_np, y_train, args.rf_cap, seed)
    log(f"训练 Random Forest  样本={len(X_rf)}  trees={args.rf_trees}  max_depth={args.rf_max_depth}  n_jobs={args.rf_jobs}")

    sw_rf = compute_sample_weight("balanced", y_rf)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators        = args.rf_trees,
        max_depth           = args.rf_max_depth,   # 限制深度，防止 OOM
        min_samples_split   = 2,
        min_samples_leaf    = 1,
        max_features        = "sqrt",
        class_weight        = "balanced_subsample",
        n_jobs              = args.rf_jobs,         # RF 独立并行数（比 XGBoost 小）
        random_state        = seed,
    )
    rf.fit(X_rf, y_rf, sample_weight=sw_rf)
    rf_time = time.time() - t0
    log(f"  RF 训练耗时: {rf_time/60:.1f} 分钟")

    joblib.dump(rf, os.path.join(args.output_dir, "random_forest_model.pkl"))
    rf_res = evaluate(rf, X_test_np, y_test, "Random Forest", le=le_target)
    evaluation_results["random_forest"] = rf_res
    training_log["random_forest"] = {"training_time_sec": round(rf_time, 1),
                                      "n_train": len(X_rf),
                                      "params": {"n_estimators": args.rf_trees,
                                                 "max_depth": args.rf_max_depth,
                                                 "n_jobs": args.rf_jobs}}

    # ── 6. KNN（限 KNN_CAP 样本）─────────────────────────────────────────────
    if not args.skip_knn:
        log("\n" + "=" * 50)
        X_knn, y_knn = stratified_subsample(X_train_np, y_train, args.knn_cap, seed)
        log(f"训练 KNN  样本={len(X_knn)}")

        t0 = time.time()
        knn = KNeighborsClassifier(
            n_neighbors = 7,
            weights     = "distance",
            algorithm   = "auto",
            n_jobs      = args.n_jobs,
        )
        knn.fit(X_knn, y_knn)
        knn_time = time.time() - t0
        log(f"  KNN 训练耗时: {knn_time/60:.1f} 分钟")

        joblib.dump(knn, os.path.join(args.output_dir, "knn_model.pkl"))
        knn_res = evaluate(knn, X_test_np, y_test, "KNN", le=le_target)
        evaluation_results["knn"] = knn_res
        training_log["knn"] = {"training_time_sec": round(knn_time, 1),
                                "n_train": len(X_knn)}
    else:
        log("跳过 KNN（--skip-knn）")

    # ── 7. SVM（限 SVM_CAP 样本，RBF 核不可跑全量）───────────────────────────
    if not args.skip_svm:
        log("\n" + "=" * 50)
        X_svm, y_svm = stratified_subsample(X_train_np, y_train, args.svm_cap, seed)
        log(f"训练 SVM（RBF kernel）  样本={len(X_svm)}")

        sw_svm = compute_sample_weight("balanced", y_svm)
        t0 = time.time()
        svm = SVC(
            C            = 10,
            kernel       = "rbf",
            gamma        = "scale",
            class_weight = "balanced",
            probability  = True,         # 支持 predict_proba / Top-K
            random_state = seed,
        )
        svm.fit(X_svm, y_svm, sample_weight=sw_svm)
        svm_time = time.time() - t0
        log(f"  SVM 训练耗时: {svm_time/60:.1f} 分钟")

        joblib.dump(svm, os.path.join(args.output_dir, "svm_model.pkl"))
        svm_res = evaluate(svm, X_test_np, y_test, "SVM", le=le_target)
        evaluation_results["svm"] = svm_res
        training_log["svm"] = {"training_time_sec": round(svm_time, 1),
                                "n_train": len(X_svm),
                                "params": {"C": 10, "kernel": "rbf"}}
    else:
        log("跳过 SVM（--skip-svm）")

    # ── 8. 汇总报告 ────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("训练完成！汇总报告：")
    log("-" * 60)
    log(f"{'模型':<16} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'F1-macro':>9} {'训练(min)':>10}")
    log("-" * 60)
    for name, res in evaluation_results.items():
        tl = training_log.get(name, {})
        t_min = tl.get("training_time_sec", 0) / 60
        top3 = f"{res['top3_accuracy']:.4f}" if res.get("top3_accuracy") else "  N/A "
        top5 = f"{res['top5_accuracy']:.4f}" if res.get("top5_accuracy") else "  N/A "
        log(f"{name:<16} {res['top1_accuracy']:>7.4f} {top3:>7} {top5:>7} "
            f"{res['f1_macro']:>9.4f} {t_min:>10.1f}")
    log("=" * 60)

    report = {
        "meta":               meta,
        "training_log":       training_log,
        "evaluation_results": evaluation_results,
    }
    report_path = os.path.join(args.output_dir, "model_evaluation_report.json")
    json.dump(report, open(report_path, "w"), indent=2, ensure_ascii=False)
    log(f"评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
