import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


UNKNOWN_SPECIES = "Unknown Stonefly"
MIN_SPECIES_COUNT = 50
UNKNOWN_CATEGORY = "__UNKNOWN__"
FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "month",
    "habitat",
    "sex",
    "life_stage",
]
NUMERIC_FEATURES = ["lat", "lon", "month"]
CATEGORICAL_FEATURES = ["country", "family", "habitat", "sex", "life_stage"]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class Data2EnhancedPreprocessor:
    """data2 增强训练专用预处理器。

    该预处理器只处理实验中表现最好的 8 个共有字段。主数据集中没有
    month/habitat/sex/life_stage 时统一补为 unknown；data2 中没有体长、颜色、
    头部特征，因此这些字段不进入本训练流程。
    """

    def __init__(self):
        self.feature_columns = FEATURES.copy()
        self.numeric_features = NUMERIC_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_fill_values = {}
        self.categorical_fill_values = {}

    def fit(self, df):
        X = df[self.feature_columns].copy()

        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            fill_value = float(X[col].median()) if not X[col].dropna().empty else 0.0
            self.numeric_fill_values[col] = fill_value
            X[col] = X[col].fillna(fill_value)

        for col in self.categorical_features:
            X[col] = X[col].fillna("unknown").astype(str).str.lower()
            mode_value = X[col].mode()[0] if not X[col].mode().empty else "unknown"
            self.categorical_fill_values[col] = mode_value
            encoder = LabelEncoder()
            encoder.fit(pd.Series(X[col].unique().tolist() + [UNKNOWN_CATEGORY]))
            self.label_encoders[col] = encoder

        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, df):
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要字段: {', '.join(missing_columns)}")

        X = df[self.feature_columns].copy()

        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(self.numeric_fill_values[col])

        for col in self.categorical_features:
            encoder = self.label_encoders[col]
            known_values = set(encoder.classes_)
            X[col] = X[col].fillna(self.categorical_fill_values[col]).astype(str).str.lower()
            X[col] = X[col].where(X[col].isin(known_values), UNKNOWN_CATEGORY)
            X[col] = encoder.transform(X[col])

        X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train four species models with main dataset + data2 enhanced features"
    )
    parser.add_argument(
        "--main-data",
        default="data/final_stonefly_dataset.csv",
        help="当前主训练 CSV 路径",
    )
    parser.add_argument(
        "--data2-occurrence",
        default="data2/occurrence.txt",
        help="data2 中的 Darwin Core occurrence.txt 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/saved_models_data2_enhanced",
        help="增强模型输出目录，默认不覆盖现有 saved_models",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="训练集最多抽样数量，0 表示使用完整训练集",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--n-jobs", type=int, default=4, help="树模型并行线程数")
    parser.add_argument(
        "--models",
        default="random_forest,svm,xgboost,knn",
        help="逗号分隔模型列表，可选 random_forest,svm,xgboost,knn",
    )
    return parser.parse_args()


def normalize_species_name(value):
    """把 `属名 种加词 作者 年份` 统一成 `属名 种加词`。"""
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = value.strip().split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def load_main_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"lifeStage": "life_stage"})
    df["month"] = "unknown"
    df["habitat"] = "unknown"
    df["sex"] = "unknown"
    df["life_stage"] = "unknown"
    df["source_dataset"] = "main"
    return df


def load_data2_dataset(path):
    occurrence = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    scientific = occurrence.get("scientificName", pd.Series("", index=occurrence.index))
    accepted = occurrence.get(
        "acceptedScientificName", pd.Series("", index=occurrence.index)
    )
    species = scientific.where(scientific.fillna("").str.strip() != "", accepted)

    df = pd.DataFrame(
        {
            "species": species.map(normalize_species_name),
            "lat": pd.to_numeric(
                occurrence.get("decimalLatitude", pd.Series(np.nan, index=occurrence.index)),
                errors="coerce",
            ),
            "lon": pd.to_numeric(
                occurrence.get("decimalLongitude", pd.Series(np.nan, index=occurrence.index)),
                errors="coerce",
            ),
            "country": occurrence.get("countryCode", "unknown"),
            "family": occurrence.get("family", "unknown"),
            "month": occurrence.get("month", "unknown"),
            "habitat": occurrence.get("habitat", "unknown"),
            "sex": occurrence.get("sex", "unknown"),
            "life_stage": occurrence.get("lifeStage", "unknown"),
            "source_dataset": "data2",
        }
    )
    df = df.replace({"": np.nan})
    df = df.dropna(subset=["species", "lat", "lon", "family"])
    df = df[df["species"] != UNKNOWN_SPECIES]
    return df.reset_index(drop=True)


def build_dataset(main_data_path, data2_occurrence_path):
    main_df = load_main_dataset(main_data_path)
    data2_df = load_data2_dataset(data2_occurrence_path)
    combined = pd.concat([main_df, data2_df], ignore_index=True)

    counts = combined["species"].value_counts()
    valid_species = counts[
        (counts >= MIN_SPECIES_COUNT) & (counts.index != UNKNOWN_SPECIES)
    ].index
    filtered = combined[combined["species"].isin(valid_species)].copy()
    filtered = filtered[filtered["species"] != UNKNOWN_SPECIES].reset_index(drop=True)

    return filtered, main_df, data2_df


def split_dataset(df, random_state):
    """按 70/15/15 划分训练、测试、验证集。"""
    y = df["species"].to_numpy()
    first_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state
    )
    train_idx, holdout_idx = next(first_split.split(np.arange(len(df)), y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    second_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )
    test_idx, validation_idx = next(
        second_split.split(np.arange(len(df_holdout)), df_holdout["species"].to_numpy())
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    df_validation = df_holdout.iloc[validation_idx].reset_index(drop=True)
    return df_train, df_test, df_validation


def sample_train(df_train, max_train_samples, random_state):
    if not max_train_samples or max_train_samples <= 0 or len(df_train) <= max_train_samples:
        return df_train.reset_index(drop=True)
    sampler = StratifiedShuffleSplit(
        n_splits=1, train_size=max_train_samples, random_state=random_state
    )
    sampled_idx, _ = next(
        sampler.split(np.arange(len(df_train)), df_train["species"].to_numpy())
    )
    return df_train.iloc[sampled_idx].reset_index(drop=True)


def top_k_accuracy(y_true, proba, class_labels, k):
    order = np.argsort(proba, axis=1)[:, -k:]
    top_labels = class_labels[order]
    return float(np.mean([y_true[i] in top_labels[i] for i in range(len(y_true))]))


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    row = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }
    if proba is not None:
        class_labels = getattr(model, "classes_", np.arange(proba.shape[1]))
        for k in [3, 4, 5]:
            row[f"top_{k}_accuracy"] = top_k_accuracy(y_test, proba, class_labels, k)
    log(
        f"{name}: Top-1={row['accuracy']:.4f}, MacroF1={row['macro_f1']:.4f}, "
        f"Top-4={row.get('top_4_accuracy', 0.0):.4f}"
    )
    return row


def train_models(model_names, X_train, y_train, X_test, y_test, random_state, n_jobs):
    models = {}
    results = {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    if "random_forest" in model_names:
        log("训练 Random Forest...")
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=399,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        results["random_forest"] = evaluate_model("random_forest", model, X_test, y_test)
        results["random_forest"]["training_time_seconds"] = time.time() - start
        models["random_forest"] = model

    if "svm" in model_names:
        log("训练 SVM...")
        start = time.time()
        model = SVC(C=10, gamma="scale", kernel="rbf", probability=True, class_weight="balanced")
        model.fit(X_train, y_train)
        results["svm"] = evaluate_model("svm", model, X_test, y_test)
        results["svm"]["training_time_seconds"] = time.time() - start
        models["svm"] = model

    if "xgboost" in model_names:
        log("训练 XGBoost...")
        start = time.time()
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        results["xgboost"] = evaluate_model("xgboost", model, X_test, y_test)
        results["xgboost"]["training_time_seconds"] = time.time() - start
        models["xgboost"] = model

    if "knn" in model_names:
        log("训练 KNN...")
        start = time.time()
        model = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=n_jobs)
        model.fit(X_train, y_train)
        results["knn"] = evaluate_model("knn", model, X_test, y_test)
        results["knn"]["training_time_seconds"] = time.time() - start
        models["knn"] = model

    return models, results


def save_outputs(output_dir, models, results, preprocessor, target_encoder, metadata, df_validation):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path / "data2_enhanced_preprocessor.pkl")
    joblib.dump(target_encoder, output_path / "target_encoder.pkl")

    for name, model in models.items():
        joblib.dump(model, output_path / f"{name}_model.pkl")

    best_model = max(results.items(), key=lambda item: item[1]["macro_f1"])[0]
    report = {
        "created_at": datetime.now().isoformat(),
        "best_model": best_model,
        "metric_used": "macro_f1",
        "features": FEATURES,
        "metadata": metadata,
        "all_results": results,
    }
    with open(output_path / "model_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_path / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    validation_records = df_validation[["species"] + FEATURES].to_dict(orient="records")
    with open(output_path / "validation_samples.json", "w", encoding="utf-8") as f:
        json.dump(validation_records, f, ensure_ascii=False, indent=2)

    log(f"模型和结果已保存到: {output_path}")
    log(f"最佳模型: {best_model}")


def main():
    args = parse_args()
    allowed_models = {"random_forest", "svm", "xgboost", "knn"}
    model_names = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown_models = sorted(set(model_names) - allowed_models)
    if unknown_models:
        raise ValueError(f"未知模型: {', '.join(unknown_models)}")

    log("读取并合并主数据集和 data2...")
    df, main_df, data2_df = build_dataset(args.main_data, args.data2_occurrence)
    df_train, df_test, df_validation = split_dataset(df, args.random_state)
    df_fit = sample_train(df_train, args.max_train_samples, args.random_state)

    log(
        f"过滤后样本={len(df)}, 类别={df['species'].nunique()}, "
        f"训练={len(df_fit)}, 测试={len(df_test)}, 验证={len(df_validation)}"
    )
    log(f"使用增强特征: {', '.join(FEATURES)}")

    preprocessor = Data2EnhancedPreprocessor()
    X_train = preprocessor.fit_transform(df_fit)
    X_test = preprocessor.transform(df_test)

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(df_fit["species"])
    y_test = target_encoder.transform(df_test["species"])

    models, results = train_models(
        model_names,
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    metadata = {
        "main_rows_loaded": int(len(main_df)),
        "data2_rows_loaded": int(len(data2_df)),
        "filtered_samples": int(len(df)),
        "class_count": int(df["species"].nunique()),
        "split_ratio": {"train": 0.70, "test": 0.15, "validation": 0.15},
        "max_train_samples": int(args.max_train_samples),
        "train_samples_used": int(len(df_fit)),
        "test_samples": int(len(df_test)),
        "validation_samples": int(len(df_validation)),
        "unknown_policy": "removed",
        "min_species_count": MIN_SPECIES_COUNT,
    }
    save_outputs(
        args.output_dir,
        models,
        results,
        preprocessor,
        target_encoder,
        metadata,
        df_validation,
    )


if __name__ == "__main__":
    main()
