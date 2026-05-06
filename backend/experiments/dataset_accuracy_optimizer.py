import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import joblib
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

BASE_FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "month",
    "habitat",
    "sex",
    "life_stage",
]

DERIVED_FEATURES = BASE_FEATURES + [
    "season",
    "lat_bin",
    "lon_bin",
    "geo_cell",
    "habitat_group",
]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search legitimate dataset/feature scopes for higher Top-1 accuracy"
    )
    parser.add_argument("--main-data", default="data/final_stonefly_dataset.csv")
    parser.add_argument("--data2-occurrence", default="data2/occurrence.txt")
    parser.add_argument(
        "--output-dir",
        default="backend/saved_models/dataset_accuracy_optimizer",
        help="实验结果输出目录",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=20000,
        help="每个候选数据集最多训练样本数，0 表示不抽样",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速筛选：只跑 XGBoost、RandomForest、KNN，不跑较慢的 SVM",
    )
    parser.add_argument(
        "--save-best-dataset",
        action="store_true",
        help="保存当前搜索到的最佳候选数据集 CSV，供下一轮正式训练使用",
    )
    return parser.parse_args()


def normalize_species_name(value):
    """统一学名格式。

    data2 的 scientificName 往往带作者和年份，当前主数据集只使用“属名 种加词”。
    为了避免同一物种被拆成两个标签，这里统一压缩成前两个单词。
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = value.strip().split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def load_main_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_dataset"] = "main"
    for col in ["month", "habitat", "sex", "life_stage"]:
        df[col] = "unknown"
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


def normalize_habitat(value):
    """把 habitat 文本压缩成较少类别，降低拼写差异带来的噪声。"""
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


def add_derived_features(df):
    enriched = df.copy()
    enriched["month"] = enriched["month"].fillna("unknown")
    enriched["season"] = enriched["month"].map(month_to_season)
    enriched["habitat"] = enriched["habitat"].fillna("unknown").astype(str).str.lower()
    enriched["habitat_group"] = enriched["habitat"].map(normalize_habitat)
    enriched["sex"] = enriched["sex"].fillna("unknown").astype(str).str.lower()
    enriched["life_stage"] = enriched["life_stage"].fillna("unknown").astype(str).str.lower()
    enriched["country"] = enriched["country"].fillna("unknown").astype(str).str.upper()
    enriched["family"] = enriched["family"].fillna("unknown").astype(str)
    enriched["lat"] = pd.to_numeric(enriched["lat"], errors="coerce")
    enriched["lon"] = pd.to_numeric(enriched["lon"], errors="coerce")

    # 经纬度网格是合法特征：用户已经提供经纬度，网格化能让树模型更稳定地学习区域分布。
    enriched["lat_bin"] = enriched["lat"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    enriched["lon_bin"] = enriched["lon"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    enriched["geo_cell"] = (
        enriched["lat_bin"].fillna("unknown").astype(str)
        + "_"
        + enriched["lon_bin"].fillna("unknown").astype(str)
    )
    return enriched.dropna(subset=["species", "lat", "lon", "family"]).reset_index(drop=True)


def filter_by_scope(df, source_scope, min_species_count, top_n_species=None):
    scoped = df.copy()
    if source_scope != "combined":
        scoped = scoped[scoped["source_dataset"] == source_scope].copy()
    scoped = scoped[scoped["species"] != UNKNOWN_SPECIES]

    counts = scoped["species"].value_counts()
    valid_species = counts[counts >= min_species_count].index
    if top_n_species:
        valid_species = counts[counts.index.isin(valid_species)].head(top_n_species).index
    return scoped[scoped["species"].isin(valid_species)].copy().reset_index(drop=True)


def split_dataset(df):
    y = df["species"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(splitter.split(np.arange(len(df)), y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    holdout_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    test_idx, _ = next(
        holdout_splitter.split(np.arange(len(df_holdout)), df_holdout["species"].to_numpy())
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


def sample_train(df_train, max_train_samples):
    if not max_train_samples or max_train_samples <= 0 or len(df_train) <= max_train_samples:
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
    numeric_features = [col for col in ["lat", "lon", "month", "lat_bin", "lon_bin"] if col in features]
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
        encoder.fit(pd.Series(X_train[col].unique().tolist() + [UNKNOWN_CATEGORY]))
        known = set(encoder.classes_)
        X_test[col] = X_test[col].where(X_test[col].isin(known), UNKNOWN_CATEGORY)
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


def run_candidate(name, df, features, args, model_names):
    if df["species"].nunique() < 2 or len(df) < 100:
        return []

    df_train, df_test = split_dataset(df)
    df_fit = sample_train(df_train, args.max_train_samples)
    X_train, X_test, y_train, y_test = prepare_matrix(df_fit, df_test, features)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    models = build_models(model_names, len(np.unique(y_train)))
    rows = []

    log(
        f"{name}: 样本={len(df)}, 类别={df['species'].nunique()}, "
        f"训练={len(df_fit)}, 测试={len(df_test)}, 特征={len(features)}"
    )
    for model_name, model in models.items():
        start = time.time()
        if model_name in {"random_forest", "xgboost"}:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {
            "candidate": name,
            "model": model_name,
            "samples": int(len(df)),
            "classes": int(df["species"].nunique()),
            "train_samples_used": int(len(df_fit)),
            "test_samples": int(len(df_test)),
            "features": features,
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "training_time_seconds": float(time.time() - start),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            class_labels = getattr(model, "classes_", np.arange(proba.shape[1]))
            for k in [3, 4, 5]:
                order = np.argsort(proba, axis=1)[:, -k:]
                top_labels = class_labels[order]
                row[f"top_{k}_accuracy"] = float(
                    np.mean([y_test[i] in top_labels[i] for i in range(len(y_test))])
                )
        log(
            f"  {model_name}: Top-1={row['accuracy']:.4f}, "
            f"MacroF1={row['macro_f1']:.4f}, Top-4={row.get('top_4_accuracy', 0.0):.4f}"
        )
        rows.append(row)
    return rows


def candidate_configs():
    """生成合法的数据集筛选方案。

    这些方案都属于明确缩小任务范围或增加真实输入特征，不包含标签泄露。
    如果某个方案达到 90%，应向客户说明适用范围，例如“只覆盖高频 20 个物种”。
    """
    for source_scope in ["combined", "data2"]:
        for min_count in [50, 100, 200, 300]:
            yield {
                "source_scope": source_scope,
                "min_species_count": min_count,
                "top_n_species": None,
                "feature_set": "derived",
                "features": DERIVED_FEATURES,
            }
        for top_n in [20, 30, 50]:
            yield {
                "source_scope": source_scope,
                "min_species_count": 50,
                "top_n_species": top_n,
                "feature_set": "derived",
                "features": DERIVED_FEATURES,
            }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["random_forest", "xgboost", "knn"] if args.quick else [
        "random_forest",
        "svm",
        "xgboost",
        "knn",
    ]

    log("读取并构造增强数据集...")
    main_df = load_main_dataset(args.main_data)
    data2_df = load_data2_dataset(args.data2_occurrence)
    combined = add_derived_features(pd.concat([main_df, data2_df], ignore_index=True))

    all_rows = []
    best_row = None
    best_dataset = None
    best_config = None

    for config in candidate_configs():
        df = filter_by_scope(
            combined,
            source_scope=config["source_scope"],
            min_species_count=config["min_species_count"],
            top_n_species=config["top_n_species"],
        )
        name = (
            f"{config['source_scope']}_min{config['min_species_count']}"
            f"_top{config['top_n_species'] or 'all'}_{config['feature_set']}"
        )
        rows = run_candidate(name, df, config["features"], args, model_names)
        all_rows.extend(rows)
        for row in rows:
            if best_row is None or row["accuracy"] > best_row["accuracy"]:
                best_row = row
                best_dataset = df
                best_config = config

    result_path = output_dir / "dataset_accuracy_optimizer_results.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "purpose": "Find legitimate dataset scopes/features that improve Top-1 accuracy",
        "warning": "Do not treat narrowed-scope results as full 124/145-class accuracy.",
        "best": best_row,
        "results": all_rows,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.save_best_dataset and best_dataset is not None:
        dataset_path = output_dir / "best_candidate_dataset.csv"
        best_dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
        with open(output_dir / "best_candidate_config.json", "w", encoding="utf-8") as f:
            json.dump(best_config, f, ensure_ascii=False, indent=2)
        log(f"最佳候选数据集已保存: {dataset_path}")

    log(f"结果已保存: {result_path}")
    if best_row:
        log(
            f"当前最佳: {best_row['candidate']} / {best_row['model']} "
            f"Top-1={best_row['accuracy']:.4f}, MacroF1={best_row['macro_f1']:.4f}, "
            f"Top-4={best_row.get('top_4_accuracy', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
