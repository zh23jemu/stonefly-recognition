import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


UNKNOWN_SPECIES = "Unknown Stonefly"
MIN_SPECIES_COUNT = 50
RANDOM_STATE = 42

BASELINE_FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "body_length_mm",
    "color",
    "head_feature",
]

ENHANCED_FEATURES = BASELINE_FEATURES + [
    "month",
    "habitat",
    "sex",
    "life_stage",
]

COMMON_ENHANCED_FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "month",
    "habitat",
    "sex",
    "life_stage",
]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate whether data2 improves Top-1")
    parser.add_argument(
        "--main-data",
        default="data/final_stonefly_dataset.csv",
        help="Current training CSV",
    )
    parser.add_argument(
        "--data2-occurrence",
        default="data2/occurrence.txt",
        help="Darwin Core occurrence.txt from data2",
    )
    parser.add_argument(
        "--output",
        default="backend/saved_models/data2_experiments/data2_feature_experiment_results.json",
        help="Result JSON path",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="Maximum stratified training samples per scenario",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=250,
        help="XGBoost estimators for quick comparison",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="XGBoost worker threads",
    )
    return parser.parse_args()


def normalize_species_name(value):
    """把带作者信息的学名压缩成“属名 种加词”。

    data2 的 scientificName 常见格式是 `Allocapnia mohri Ross & Ricker, 1964`，
    当前主数据集使用 `Allocapnia mohri`，所以实验合并前需要统一命名口径。
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = value.strip().split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def load_main_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"lifeStage": "life_stage"})
    df["source_dataset"] = "main"
    df["month"] = "unknown"
    df["habitat"] = "unknown"
    df["sex"] = "unknown"
    df["life_stage"] = "unknown"
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
            # data2 没有体长、颜色、头部特征。这里保留列位，后续统一填补，
            # 用来测试“直接合并 data2”是否真的有收益。
            "body_length_mm": np.nan,
            "color": "unknown",
            "head_feature": "unknown",
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


def filter_known_species(df):
    counts = df["species"].value_counts()
    valid_species = counts[(counts >= MIN_SPECIES_COUNT) & (counts.index != UNKNOWN_SPECIES)].index
    return df[df["species"].isin(valid_species)].copy().reset_index(drop=True)


def split_train_test(df):
    """沿用正式训练的 70/15/15 口径，只取测试集做实验对比。"""
    y = df["species"].to_numpy()
    first_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RANDOM_STATE
    )
    train_idx, holdout_idx = next(first_split.split(np.arange(len(df)), y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    second_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    test_idx, _ = next(
        second_split.split(np.arange(len(df_holdout)), df_holdout["species"].to_numpy())
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


def sample_train(df_train, max_train_samples):
    if not max_train_samples or len(df_train) <= max_train_samples:
        return df_train
    sampler = StratifiedShuffleSplit(
        n_splits=1, train_size=max_train_samples, random_state=RANDOM_STATE
    )
    sampled_idx, _ = next(
        sampler.split(np.arange(len(df_train)), df_train["species"].to_numpy())
    )
    return df_train.iloc[sampled_idx].reset_index(drop=True)


def prepare_matrix(df_train, df_test, features):
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    numeric_features = [
        col for col in ["lat", "lon", "body_length_mm", "month"] if col in features
    ]
    categorical_features = [col for col in features if col not in numeric_features]

    for col in numeric_features:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        fill_value = float(X_train[col].median()) if not X_train[col].dropna().empty else 0.0
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    for col in categorical_features:
        X_train[col] = X_train[col].fillna("unknown").astype(str).str.lower()
        X_test[col] = X_test[col].fillna("unknown").astype(str).str.lower()
        encoder = LabelEncoder()
        encoder.fit(pd.Series(X_train[col].unique().tolist() + ["__unknown__"]))
        known = set(encoder.classes_)
        X_test[col] = X_test[col].where(X_test[col].isin(known), "__unknown__")
        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    if numeric_features:
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(df_train["species"])
    y_test = target_encoder.transform(df_test["species"])
    return X_train, X_test, y_train, y_test


def top_k_accuracy(y_true, proba, class_labels, k):
    order = np.argsort(proba, axis=1)[:, -k:]
    top_labels = class_labels[order]
    return float(np.mean([y_true[i] in top_labels[i] for i in range(len(y_true))]))


def run_scenario(name, df, features, args):
    df = filter_known_species(df)
    df_train, df_test = split_train_test(df)
    df_fit = sample_train(df_train, args.max_train_samples)
    X_train, X_test, y_train, y_test = prepare_matrix(df_fit, df_test, features)

    log(
        f"{name}: 样本={len(df)}, 类别={df['species'].nunique()}, "
        f"训练={len(df_fit)}, 测试={len(df_test)}, 特征={len(features)}"
    )
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    class_labels = getattr(model, "classes_", np.arange(proba.shape[1]))

    return {
        "scenario": name,
        "samples": int(len(df)),
        "classes": int(df["species"].nunique()),
        "train_samples_used": int(len(df_fit)),
        "test_samples": int(len(df_test)),
        "features": features,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "top_3_accuracy": top_k_accuracy(y_test, proba, class_labels, 3),
        "top_4_accuracy": top_k_accuracy(y_test, proba, class_labels, 4),
        "top_5_accuracy": top_k_accuracy(y_test, proba, class_labels, 5),
    }


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log("读取当前数据集和 data2...")
    main_df = load_main_dataset(args.main_data)
    data2_df = load_data2_dataset(args.data2_occurrence)
    combined_df = pd.concat([main_df, data2_df], ignore_index=True)

    scenarios = [
        (
            "main_baseline_7_features",
            main_df,
            BASELINE_FEATURES,
        ),
        (
            "main_plus_data2_baseline_7_features",
            combined_df,
            BASELINE_FEATURES,
        ),
        (
            "main_plus_data2_enhanced_11_features",
            combined_df,
            ENHANCED_FEATURES,
        ),
        (
            "main_plus_data2_common_enhanced_8_features",
            combined_df,
            COMMON_ENHANCED_FEATURES,
        ),
    ]

    results = []
    for name, df, features in scenarios:
        results.append(run_scenario(name, df, features, args))

    payload = {
        "created_at": datetime.now().isoformat(),
        "purpose": "Evaluate whether data2 occurrence fields improve species Top-1",
        "data2_rows_loaded": int(len(data2_df)),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log(f"结果已保存: {output_path}")
    for row in results:
        log(
            f"{row['scenario']}: Top-1={row['accuracy']:.4f}, "
            f"MacroF1={row['macro_f1']:.4f}, Top-4={row['top_4_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
