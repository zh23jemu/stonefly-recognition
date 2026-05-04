import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

backend_root = os.path.dirname(os.path.dirname(__file__))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)


DATASET_PATH = Path(backend_root).parent / "data" / "final_stonefly_dataset.csv"
DEFAULT_PROGRESS_DIR = Path(backend_root) / "saved_models" / "model_experiments"
DEFAULT_STATE_FILE = DEFAULT_PROGRESS_DIR / "catboost_lightgbm_ensemble_state.json"
DEFAULT_RESULT_FILE = DEFAULT_PROGRESS_DIR / "catboost_lightgbm_ensemble_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="CatBoost + LightGBM + ensemble sweep")
    parser.add_argument(
        "--data-path",
        default=str(DATASET_PATH),
        help="Path to the raw CSV dataset",
    )
    parser.add_argument("--cv", type=int, default=3, help="Reserved for future use")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="Maximum stratified samples used from the training split",
    )
    parser.add_argument(
        "--progress-dir",
        default=str(DEFAULT_PROGRESS_DIR),
        help="Directory for progress and result files",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="File to persist checkpoint state",
    )
    parser.add_argument(
        "--result-file",
        default=str(DEFAULT_RESULT_FILE),
        help="Final result JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing state file if present",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing state and start from scratch",
    )
    return parser.parse_args()


def log(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_dict(name, y_true, y_pred, proba, labels):
    row = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None:
        for k in [3, 4, 5]:
            row[f"top_{k}_accuracy"] = float(
                top_k_accuracy_score(y_true, proba, k=k, labels=labels)
            )
    return row


def build_dataset(data_path):
    log("读取并划分数据...")
    df = pd.read_csv(data_path, encoding="utf-8")
    unknown_label = "Unknown Stonefly"
    counts = df["species"].value_counts()
    valid = counts[(counts >= 50) & (counts.index != unknown_label)].index
    df = df[df["species"].isin(valid) & (df["species"] != unknown_label)].copy().reset_index(drop=True)

    y = df["species"].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, holdout_idx = next(sss.split(np.arange(len(df)), y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    test_idx, _ = next(sss2.split(np.arange(len(df_holdout)), df_holdout["species"].to_numpy()))
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)

    if len(df_train) > 20000:
        sss3 = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=42)
        sample_idx, _ = next(sss3.split(np.arange(len(df_train)), df_train["species"].to_numpy()))
        df_fit = df_train.iloc[sample_idx].reset_index(drop=True)
    else:
        df_fit = df_train

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_fit["species"])
    y_test = label_encoder.transform(df_test["species"])
    labels = np.arange(len(label_encoder.classes_))

    features = ["lat", "lon", "country", "family", "body_length_mm", "color", "head_feature"]
    cat_features = ["country", "family", "color", "head_feature"]
    num_features = ["lat", "lon", "body_length_mm"]

    # 统一序数编码，保证与当前线上接口一致；CatBoost单独保留原始分类特征输入。
    X_train = df_fit[features].copy()
    X_test = df_test[features].copy()
    encoders = {}
    for col in cat_features:
        enc = LabelEncoder()
        enc.fit(list(X_train[col].astype(str).unique()) + ["__UNKNOWN__"])
        known = set(enc.classes_)
        X_train[col] = enc.transform(
            X_train[col].astype(str).where(X_train[col].astype(str).isin(known), "__UNKNOWN__")
        )
        X_test[col] = enc.transform(
            X_test[col].astype(str).where(X_test[col].astype(str).isin(known), "__UNKNOWN__")
        )
        encoders[col] = enc

    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    X_train_cat = df_fit[features].copy()
    X_test_cat = df_test[features].copy()
    for col in cat_features:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)
    cat_indices = [features.index(c) for c in cat_features]

    dataset = {
        "df": df,
        "df_fit": df_fit,
        "df_test": df_test,
        "y_train": y_train,
        "y_test": y_test,
        "labels": labels,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_cat": X_train_cat,
        "X_test_cat": X_test_cat,
        "cat_indices": cat_indices,
        "features": features,
        "unknown_label": unknown_label,
    }
    return dataset


def load_state(state_file: Path):
    state = load_json(state_file)
    if not state:
        return {
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "results": [],
        }
    return state


def persist_state(state_file: Path, state: dict):
    save_json(state_file, state)


def maybe_append_result(state, row):
    state["results"].append(row)
    state["completed"].append(row["model"])


def run_single_model(name, model, train_x, test_x, y_train, y_test, labels, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    log(f"开始训练 {name} ...")
    start = time.time()
    if name == "catboost":
        model.fit(train_x, y_train, **fit_kwargs)
    else:
        model.fit(train_x, y_train)
    elapsed = time.time() - start
    pred = np.asarray(model.predict(test_x)).reshape(-1).astype(int)
    proba = model.predict_proba(test_x)
    row = metric_dict(name, y_test, pred, proba, labels)
    row["training_time_seconds"] = float(elapsed)
    log(
        f"{name} 完成: acc={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
        f"top4={row.get('top_4_accuracy', 0.0):.4f}, 用时={elapsed/60:.1f} 分钟"
    )
    return row, model


def run(args):
    progress_dir = Path(args.progress_dir)
    state_file = Path(args.state_file)
    result_file = Path(args.result_file)
    progress_dir.mkdir(parents=True, exist_ok=True)

    if args.force_restart and state_file.exists():
        state_file.unlink()

    state = load_state(state_file) if args.resume or state_file.exists() else {
        "started_at": datetime.now().isoformat(),
        "completed": [],
        "results": [],
    }

    dataset = build_dataset(args.data_path)
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    labels = dataset["labels"]

    log(
        f"数据准备完成: 样本={len(dataset['df'])}, 训练样本={len(dataset['df_fit'])}, 测试样本={len(dataset['df_test'])}"
    )

    model_specs = [
        (
            "catboost",
            CatBoostClassifier(
                iterations=350,
                depth=8,
                learning_rate=0.08,
                loss_function="MultiClass",
                random_seed=42,
                verbose=50,
                thread_count=2,
                allow_writing_files=False,
            ),
            dataset["X_train_cat"],
            dataset["X_test_cat"],
            {"cat_features": dataset["cat_indices"]},
        ),
        (
            "lightgbm",
            LGBMClassifier(
                objective="multiclass",
                n_estimators=350,
                learning_rate=0.06,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                n_jobs=2,
                verbose=20,
            ),
            dataset["X_train"],
            dataset["X_test"],
            {},
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=350,
                max_depth=8,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=2,
                tree_method="hist",
            ),
            dataset["X_train"],
            dataset["X_test"],
            {},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=42,
            ),
            dataset["X_train"],
            dataset["X_test"],
            {},
        ),
    ]

    fitted = {}
    for name, model, train_x, test_x, fit_kwargs in model_specs:
        if name in state["completed"]:
            log(f"跳过已完成模型: {name}")
            continue

        row, fitted_model = run_single_model(
            name,
            model,
            train_x,
            test_x,
            y_train,
            y_test,
            labels,
            fit_kwargs=fit_kwargs,
        )
        row["status"] = "done"
        maybe_append_result(state, row)
        persist_state(state_file, state)
        fitted[name] = fitted_model

        if name in {"catboost", "lightgbm", "xgboost"}:
            model_path = progress_dir / f"{name}_checkpoint.pkl"
            import joblib

            joblib.dump(fitted_model, model_path)
            log(f"已保存检查点: {model_path.name}")

    # 如果三大模型都完成了，再构建集成结果。
    required = {"catboost", "lightgbm", "xgboost"}
    if required.issubset(set(state["completed"])):
        done_models = {}
        for row in state["results"]:
            if row["model"] in required:
                done_models[row["model"]] = row

        log("开始构建软投票集成...")
        cat_model = fitted.get("catboost")
        lgb_model = fitted.get("lightgbm")
        xgb_model = fitted.get("xgboost")

        if cat_model is None:
            import joblib

            cat_model = joblib.load(progress_dir / "catboost_checkpoint.pkl")
        if lgb_model is None:
            import joblib

            lgb_model = joblib.load(progress_dir / "lightgbm_checkpoint.pkl")
        if xgb_model is None:
            import joblib

            xgb_model = joblib.load(progress_dir / "xgboost_checkpoint.pkl")

        proba_mix = np.mean(
            [
                cat_model.predict_proba(dataset["X_test_cat"]),
                lgb_model.predict_proba(dataset["X_test"]),
                xgb_model.predict_proba(dataset["X_test"]),
            ],
            axis=0,
        )
        pred_mix = np.argmax(proba_mix, axis=1)
        ensemble_row = metric_dict(
            "soft_voting_cat_lgbm_xgb",
            y_test,
            pred_mix,
            proba_mix,
            labels,
        )
        ensemble_row["status"] = "done"
        ensemble_row["training_time_seconds"] = 0.0
        maybe_append_result(state, ensemble_row)
        persist_state(state_file, state)
        log(
            f"集成完成: acc={ensemble_row['accuracy']:.4f}, macro_f1={ensemble_row['macro_f1']:.4f}, "
            f"top4={ensemble_row.get('top_4_accuracy', 0.0):.4f}"
        )

    final = {
        "started_at": state["started_at"],
        "finished_at": datetime.now().isoformat(),
        "dataset": {
            "samples": int(len(dataset["df"])),
            "classes": int(dataset["df"]["species"].nunique()),
            "train_samples_used": int(len(dataset["df_fit"])),
            "test_samples": int(len(dataset["df_test"])),
            "unknown_policy": "removed",
            "max_train_samples": int(args.max_train_samples),
        },
        "results": state["results"],
    }
    save_json(result_file, final)
    log(f"结果已保存到: {result_file}")
    log(f"进度文件: {state_file}")


if __name__ == "__main__":
    run(parse_args())
