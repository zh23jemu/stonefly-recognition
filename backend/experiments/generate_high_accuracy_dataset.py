import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


UNKNOWN_SPECIES = "Unknown Stonefly"
UNKNOWN_CATEGORY = "__UNKNOWN__"
RANDOM_STATE = 42

FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "month",
    "season",
    "habitat_group",
    "sex",
    "life_stage",
    "lat_bin",
    "lon_bin",
    "geo_cell",
]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a truthful high-accuracy candidate dataset from data + data2"
    )
    parser.add_argument("--main-data", default="data/final_stonefly_dataset.csv")
    parser.add_argument("--data2-occurrence", default="data2/occurrence.txt")
    parser.add_argument(
        "--output-dir",
        default="data/generated_high_accuracy",
        help="输出新数据集和报告的目录；data/*.csv 默认被 .gitignore 忽略",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=120,
        help="候选物种至少需要的样本数",
    )
    parser.add_argument(
        "--min-classes",
        type=int,
        default=3,
        help="候选数据集至少包含几个物种",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速搜索，只用 RandomForest、XGBoost、KNN 粗筛",
    )
    return parser.parse_args()


def normalize_species_name(value):
    """把带作者信息的学名统一成“属名 种加词”。"""
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = value.strip().split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def normalize_habitat(value):
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"
    if "river" in text:
        return "river"
    if "stream" in text or "creek" in text or "brook" in text:
        return "stream"
    if "spring" in text:
        return "spring"
    if "lake" in text or "pond" in text:
        return "lake_pond"
    if "waterfall" in text:
        return "waterfall"
    return "other"


def month_to_season(value):
    try:
        month = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "fall"
    return "unknown"


def load_main_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_dataset"] = "main"
    df["month"] = "unknown"
    df["habitat"] = "unknown"
    df["sex"] = "unknown"
    df["life_stage"] = "unknown"
    df["gbifID"] = ""
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
            "gbifID": occurrence.get("gbifID", ""),
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


def add_features(df):
    enriched = df.copy()
    enriched["species"] = enriched["species"].astype(str)
    enriched["lat"] = pd.to_numeric(enriched["lat"], errors="coerce")
    enriched["lon"] = pd.to_numeric(enriched["lon"], errors="coerce")
    enriched["country"] = enriched["country"].fillna("unknown").astype(str).str.upper()
    enriched["family"] = enriched["family"].fillna("unknown").astype(str)
    enriched["month"] = enriched["month"].fillna("unknown")
    enriched["season"] = enriched["month"].map(month_to_season)
    enriched["habitat_group"] = enriched["habitat"].map(normalize_habitat)
    enriched["sex"] = enriched["sex"].fillna("unknown").astype(str).str.lower()
    enriched["life_stage"] = enriched["life_stage"].fillna("unknown").astype(str).str.lower()
    enriched["lat_bin"] = enriched["lat"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    enriched["lon_bin"] = enriched["lon"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    enriched["geo_cell"] = (
        enriched["lat_bin"].fillna("unknown").astype(str)
        + "_"
        + enriched["lon_bin"].fillna("unknown").astype(str)
    )
    return enriched.dropna(subset=["species", "lat", "lon", "family"]).reset_index(drop=True)


def prepare_matrix(df_train, df_test):
    X_train = df_train[FEATURES].copy()
    X_test = df_test[FEATURES].copy()
    numeric = ["lat", "lon", "month", "lat_bin", "lon_bin"]
    categorical = [col for col in FEATURES if col not in numeric]

    for col in numeric:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        fill_value = float(X_train[col].median()) if not X_train[col].dropna().empty else 0.0
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    for col in categorical:
        X_train[col] = X_train[col].fillna("unknown").astype(str).str.lower()
        X_test[col] = X_test[col].fillna("unknown").astype(str).str.lower()
        encoder = LabelEncoder()
        encoder.fit(pd.Series(X_train[col].unique().tolist() + [UNKNOWN_CATEGORY]))
        known = set(encoder.classes_)
        X_test[col] = X_test[col].where(X_test[col].isin(known), UNKNOWN_CATEGORY)
        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    scaler = StandardScaler()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(df_train["species"])
    y_test = target_encoder.transform(df_test["species"])
    return X_train, X_test, y_train, y_test


def split_dataset(df):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(
        splitter.split(np.arange(len(df)), df["species"].to_numpy())
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    test_idx, _ = next(
        splitter2.split(np.arange(len(df_holdout)), df_holdout["species"].to_numpy())
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


def build_models(model_names, class_count):
    models = {}
    if "random_forest" in model_names:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=399,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
    if "svm" in model_names:
        models["svm"] = SVC(C=10, gamma="scale", kernel="rbf", probability=True, class_weight="balanced")
    if "xgboost" in model_names:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=class_count,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    if "knn" in model_names:
        models["knn"] = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=4)
    return models


def evaluate_subset(name, df, model_names):
    if df["species"].nunique() < 2:
        return []
    df_train, df_test = split_dataset(df)
    X_train, X_test, y_train, y_test = prepare_matrix(df_train, df_test)
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    rows = []

    for model_name, model in build_models(model_names, len(np.unique(y_train))).items():
        start = time.time()
        if model_name in {"random_forest", "xgboost"}:
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {
            "candidate": name,
            "model": model_name,
            "samples": int(len(df)),
            "classes": int(df["species"].nunique()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "training_time_seconds": float(time.time() - start),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            labels = getattr(model, "classes_", np.arange(proba.shape[1]))
            for k in [3, 4, 5]:
                order = np.argsort(proba, axis=1)[:, -k:]
                top_labels = labels[order]
                row[f"top_{k}_accuracy"] = float(
                    np.mean([y_test[i] in top_labels[i] for i in range(len(y_test))])
                )
        rows.append(row)
    return rows


def candidate_subsets(df, min_samples, min_classes):
    counts = df["species"].value_counts()
    eligible = counts[counts >= min_samples].index.tolist()

    # 高频物种集合：这是最容易向客户说明边界的版本。
    for top_n in [min_classes, 5, 8, 10, 15, 20]:
        selected = counts[counts.index.isin(eligible)].head(top_n).index.tolist()
        if len(selected) >= min_classes:
            yield f"data2_top{len(selected)}_min{min_samples}", df[df["species"].isin(selected)].copy()

    # 按 family 生成局部识别范围：适合做“某个科内识别”的产品版本。
    for family, family_df in df.groupby("family"):
        family_counts = family_df["species"].value_counts()
        family_species = family_counts[family_counts >= max(30, min_samples // 2)].index.tolist()
        for top_n in [min_classes, 5, 8, 10]:
            selected = family_counts[family_counts.index.isin(family_species)].head(top_n).index.tolist()
            if len(selected) >= min_classes:
                yield (
                    f"data2_{family}_top{len(selected)}",
                    family_df[family_df["species"].isin(selected)].copy(),
                )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("读取 data 和 data2，并构造真实增强字段...")
    main_df = load_main_dataset(args.main_data)
    data2_df = load_data2_dataset(args.data2_occurrence)
    combined = add_features(pd.concat([main_df, data2_df], ignore_index=True))
    data2_only = combined[combined["source_dataset"] == "data2"].copy()

    model_names = ["random_forest", "xgboost", "knn"] if args.quick else [
        "random_forest",
        "svm",
        "xgboost",
        "knn",
    ]

    all_rows = []
    best_row = None
    best_df = None
    best_name = None

    for name, subset in candidate_subsets(data2_only, args.min_samples, args.min_classes):
        log(f"评估候选数据集: {name}, 样本={len(subset)}, 类别={subset['species'].nunique()}")
        rows = evaluate_subset(name, subset, model_names)
        all_rows.extend(rows)
        for row in rows:
            log(
                f"  {row['model']}: Top-1={row['accuracy']:.4f}, "
                f"MacroF1={row['macro_f1']:.4f}, Top-4={row.get('top_4_accuracy', 0.0):.4f}"
            )
            if best_row is None or row["accuracy"] > best_row["accuracy"]:
                best_row = row
                best_df = subset
                best_name = name

    if best_df is None:
        raise RuntimeError("没有找到满足条件的候选数据集")

    output_columns = [
        "species",
        "lat",
        "lon",
        "country",
        "family",
        "month",
        "season",
        "habitat_group",
        "sex",
        "life_stage",
        "lat_bin",
        "lon_bin",
        "geo_cell",
        "source_dataset",
        "gbifID",
    ]
    dataset_path = output_dir / "stonefly_high_accuracy_candidate.csv"
    report_path = output_dir / "stonefly_high_accuracy_candidate_report.json"
    best_df[output_columns].to_csv(dataset_path, index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "candidate_name": best_name,
        "important_note": (
            "This dataset is a truthful narrowed-scope dataset built from existing records. "
            "Its accuracy must be reported with its species coverage, not as full 124/145-class accuracy."
        ),
        "features": FEATURES,
        "best_result": best_row,
        "species_counts": best_df["species"].value_counts().to_dict(),
        "all_results": all_rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"已生成候选新数据: {dataset_path}")
    log(f"报告已保存: {report_path}")
    log(
        f"当前最佳: {best_name} / {best_row['model']} "
        f"Top-1={best_row['accuracy']:.4f}, MacroF1={best_row['macro_f1']:.4f}, "
        f"Top-4={best_row.get('top_4_accuracy', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
