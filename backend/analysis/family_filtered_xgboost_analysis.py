"""
评估“用户已选 family 后，平铺 XGBoost 过滤重排”的效果，并按 family 做误差拆解。

用途：
1. 读取训练产物目录中的 XGBoost、预处理器和切分元信息
2. 用与训练阶段一致的随机种子重建 test / validation 切分
3. 计算平铺 XGBoost 的原始 Top-K 指标
4. 计算“已知真实 family 后，仅在该 family 的 species 候选中重排”的 Top-K 指标
5. 输出按 family 的效果对比与主要混淆对

默认针对增强版模型目录 `backend/saved_models_augmented/`，但也尽量兼容旧版 `saved_models/`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.augmented_feature_utils import prepare_augmented_dataframe


TARGET = "species"
CANONICAL_FEATURE_ORDER = [
    "lat",
    "lon",
    "country",
    "family",
    "body_length_mm",
    "color",
    "head_feature",
    "month",
    "season",
    "habitat",
    "habitat_group",
    "sex",
    "life_stage",
    "lat_bin",
    "lon_bin",
    "geo_cell",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估已知 family 后平铺 XGBoost 的过滤重排效果，并输出按 family 的误差拆解"
    )
    parser.add_argument(
        "--models-dir",
        default=str(BACKEND_DIR / "saved_models_augmented"),
        help="训练产物目录，默认 backend/saved_models_augmented",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="原始 CSV 路径；不传时优先读取 dataset_split_metadata.json 中的 data_path",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="分析使用的切分，默认 test，与训练日志中的主评估口径一致",
    )
    parser.add_argument(
        "--top-confusions-per-family",
        type=int,
        default=3,
        help="每个 family 输出多少条最常见的错误混淆对，默认 3",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="分析结果输出目录；默认写入 <models-dir>/family_filtered_analysis",
    )
    return parser.parse_args()


def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return (base_dir / candidate).resolve()


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def infer_feature_columns(models_dir: Path, preprocessor) -> list[str]:
    """优先从 validation_samples.json 推断训练输入顺序，避免特征顺序漂移。"""
    validation_samples_path = models_dir / "validation_samples.json"
    if validation_samples_path.exists():
        validation_samples = load_json(validation_samples_path)
        if validation_samples:
            return [key for key in validation_samples[0].keys() if key != TARGET]

    feature_columns = getattr(preprocessor, "feature_columns", None)
    if feature_columns:
        return list(feature_columns)

    known_feature_set = set(getattr(preprocessor, "numeric_features", [])) | set(
        getattr(preprocessor, "categorical_features", [])
    )
    return [column for column in CANONICAL_FEATURE_ORDER if column in known_feature_set]


def resolve_data_path(args, models_dir: Path, metadata: dict) -> Path:
    if args.data_path:
        return resolve_path(args.data_path, REPO_ROOT)

    metadata_data_path = metadata.get("data_path")
    if metadata_data_path:
        # 训练脚本通常从 backend 目录运行，因此相对路径按 backend 解析最稳妥。
        return resolve_path(metadata_data_path, BACKEND_DIR)

    if "augmented" in models_dir.name:
        return REPO_ROOT / "data" / "stonefly_combined_data_augmented.csv"
    return REPO_ROOT / "data" / "final_stonefly_dataset.csv"


def load_filtered_dataset(
    data_path: Path,
    feature_columns: list[str],
    min_species_count: int,
    metadata: dict,
) -> pd.DataFrame:
    dataset = pd.read_csv(data_path, low_memory=False)
    if "month" in dataset.columns or "habitat_group" in dataset.columns:
        dataset = prepare_augmented_dataframe(dataset)

    unknown_policy = metadata.get("unknown_policy")
    unknown_label = metadata.get("unknown_label", "Unknown Stonefly")
    if unknown_policy == "removed":
        dataset = dataset[dataset[TARGET] != unknown_label].copy()

    species_counts = dataset[TARGET].value_counts()
    valid_species = species_counts[species_counts >= min_species_count].index
    dataset = dataset[dataset[TARGET].isin(valid_species)].reset_index(drop=True)

    missing_columns = [column for column in feature_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"数据集中缺少训练所需字段: {', '.join(missing_columns)}"
        )

    keep_columns = feature_columns + [TARGET]
    return dataset[keep_columns].copy()


def split_dataset(df: pd.DataFrame, seed: int, split_name: str) -> pd.DataFrame:
    """严格复用 train_augmented.py 的 70/15/15 分层切分逻辑。"""
    all_labels = df[TARGET].to_numpy()
    first_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(first_splitter.split(np.zeros(len(all_labels)), all_labels))
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    holdout_labels = df_holdout[TARGET].to_numpy()
    second_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    test_idx, validation_idx = next(
        second_splitter.split(np.zeros(len(holdout_labels)), holdout_labels)
    )

    if split_name == "test":
        return df_holdout.iloc[test_idx].reset_index(drop=True)
    return df_holdout.iloc[validation_idx].reset_index(drop=True)


def load_xgboost_species_labels(models_dir: Path, preprocessor, model) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("XGBoost 模型缺少 classes_，无法构造 species 标签映射")

    classes = np.asarray(classes).astype(int)
    xgb_label_encoder_path = models_dir / "xgboost_label_encoder.pkl"
    if xgb_label_encoder_path.exists():
        xgb_label_encoder = joblib.load(xgb_label_encoder_path)
        encoded_species = xgb_label_encoder.inverse_transform(classes)
    else:
        encoded_species = classes

    return preprocessor.target_encoder.inverse_transform(np.asarray(encoded_species).astype(int))


def topk_hits(top_predictions: np.ndarray, true_species: np.ndarray, k: int) -> np.ndarray:
    return np.array(
        [true_species[index] in top_predictions[index, :k] for index in range(len(true_species))],
        dtype=bool,
    )


def build_confusion_summary(
    true_species: np.ndarray,
    pred_species: np.ndarray,
    limit: int,
) -> list[dict]:
    error_rows = pd.DataFrame(
        {"true_species": true_species, "pred_species": pred_species}
    )
    error_rows = error_rows[error_rows["true_species"] != error_rows["pred_species"]]
    if error_rows.empty:
        return []

    grouped = (
        error_rows.groupby(["true_species", "pred_species"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "true_species", "pred_species"], ascending=[False, True, True])
    )
    return grouped.head(limit).to_dict(orient="records")


def main():
    args = parse_args()
    models_dir = resolve_path(args.models_dir, BACKEND_DIR)
    if models_dir is None or not models_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {args.models_dir}")

    output_dir = (
        resolve_path(args.output_dir, BACKEND_DIR)
        if args.output_dir
        else models_dir / "family_filtered_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    xgboost_model = joblib.load(models_dir / "xgboost_model.pkl")

    metadata_path = models_dir / "dataset_split_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    seed = int(metadata.get("seed", metadata.get("random_state", 42)))
    min_species_count = int(
        metadata.get("min_species_count", metadata.get("min_class_count", 50))
    )

    feature_columns = infer_feature_columns(models_dir, preprocessor)
    data_path = resolve_data_path(args, models_dir, metadata)
    if not data_path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")

    dataset = load_filtered_dataset(data_path, feature_columns, min_species_count, metadata)
    split_df = split_dataset(dataset, seed=seed, split_name=args.split)
    species_to_family = (
        dataset.groupby(TARGET)["family"].agg(lambda values: values.mode().iat[0]).to_dict()
    )

    x_matrix, y_true_encoded, _ = preprocessor.transform_pipeline(split_df, TARGET)
    if hasattr(x_matrix, "to_numpy"):
        x_input = x_matrix.to_numpy(dtype=np.float32)
    else:
        x_input = np.asarray(x_matrix, dtype=np.float32)

    true_species = preprocessor.target_encoder.inverse_transform(y_true_encoded)
    true_families = split_df["family"].to_numpy()

    probabilities = xgboost_model.predict_proba(x_input)
    class_species = load_xgboost_species_labels(models_dir, preprocessor, xgboost_model)
    class_families = np.array(
        [species_to_family.get(species, "__UNKNOWN_FAMILY__") for species in class_species]
    )

    max_k = 5
    flat_order = np.argsort(probabilities, axis=1)[:, ::-1]
    flat_top_species = class_species[flat_order[:, :max_k]]
    flat_top1_predictions = flat_top_species[:, 0]

    family_to_class_indices = {
        family_name: np.where(class_families == family_name)[0]
        for family_name in sorted(np.unique(class_families))
    }

    filtered_top_species = np.empty((len(split_df), max_k), dtype=object)
    filtered_top_species[:] = ""
    filtered_top1_predictions = np.empty(len(split_df), dtype=object)
    per_family_rows = []
    overall_filtered_hits = {k: np.zeros(len(split_df), dtype=bool) for k in [1, 3, 4, 5]}
    overall_flat_hits = {
        k: topk_hits(flat_top_species, true_species, k) for k in [1, 3, 4, 5]
    }

    for family_name in sorted(pd.unique(true_families)):
        row_indices = np.where(true_families == family_name)[0]
        class_indices = family_to_class_indices.get(family_name)
        if class_indices is None or len(class_indices) == 0:
            raise ValueError(f"找不到 family={family_name} 对应的 species 候选集合")

        family_probabilities = probabilities[row_indices][:, class_indices]
        family_species = class_species[class_indices]
        family_order = np.argsort(family_probabilities, axis=1)[:, ::-1]
        ordered_family_species = family_species[family_order[:, :max_k]]

        filtered_top_species[row_indices, : ordered_family_species.shape[1]] = ordered_family_species
        filtered_top1_predictions[row_indices] = ordered_family_species[:, 0]

        family_true_species = true_species[row_indices]
        family_flat_top_species = flat_top_species[row_indices]
        family_flat_top1 = flat_top1_predictions[row_indices]
        family_filtered_top1 = filtered_top1_predictions[row_indices]

        family_metrics = {
            "family": family_name,
            "sample_count": int(len(row_indices)),
            "species_count": int(split_df.iloc[row_indices][TARGET].nunique()),
            "candidate_species_count": int(len(class_indices)),
        }

        for k in [1, 3, 4, 5]:
            filtered_hits = topk_hits(ordered_family_species, family_true_species, k)
            flat_hits = topk_hits(family_flat_top_species, family_true_species, k)
            overall_filtered_hits[k][row_indices] = filtered_hits
            family_metrics[f"flat_top{k}"] = round(float(np.mean(flat_hits)), 4)
            family_metrics[f"filtered_top{k}"] = round(float(np.mean(filtered_hits)), 4)
            family_metrics[f"top{k}_lift"] = round(
                family_metrics[f"filtered_top{k}"] - family_metrics[f"flat_top{k}"], 4
            )

        family_metrics["flat_top1_confusions"] = build_confusion_summary(
            family_true_species, family_flat_top1, args.top_confusions_per_family
        )
        family_metrics["filtered_top1_confusions"] = build_confusion_summary(
            family_true_species, family_filtered_top1, args.top_confusions_per_family
        )
        per_family_rows.append(family_metrics)

    overall_summary = {
        "split": args.split,
        "samples": int(len(split_df)),
        "species_count": int(split_df[TARGET].nunique()),
        "family_count": int(split_df["family"].nunique()),
        "flat_xgboost": {
            f"top{k}": round(float(np.mean(overall_flat_hits[k])), 4) for k in [1, 3, 4, 5]
        },
        "family_filtered_xgboost": {
            f"top{k}": round(float(np.mean(overall_filtered_hits[k])), 4)
            for k in [1, 3, 4, 5]
        },
    }
    overall_summary["lift"] = {
        f"top{k}": round(
            overall_summary["family_filtered_xgboost"][f"top{k}"]
            - overall_summary["flat_xgboost"][f"top{k}"],
            4,
        )
        for k in [1, 3, 4, 5]
    }

    family_summary_df = pd.DataFrame(per_family_rows).sort_values(
        ["filtered_top1", "top1_lift", "sample_count"],
        ascending=[False, False, False],
    )
    family_summary_df.to_csv(
        output_dir / f"family_breakdown_{args.split}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "models_dir": str(models_dir),
        "data_path": str(data_path),
        "split": args.split,
        "seed": seed,
        "min_species_count": min_species_count,
        "feature_columns": feature_columns,
        "overall_summary": overall_summary,
        "per_family": per_family_rows,
    }
    with open(
        output_dir / f"family_filtered_xgboost_report_{args.split}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Family-Filtered XGBoost Analysis")
    print("=" * 70)
    print(f"models_dir : {models_dir}")
    print(f"data_path  : {data_path}")
    print(f"split      : {args.split}")
    print(f"samples    : {overall_summary['samples']}")
    print(
        "flat       : "
        + ", ".join(
            f"Top-{k}={overall_summary['flat_xgboost'][f'top{k}']:.4f}"
            for k in [1, 3, 4, 5]
        )
    )
    print(
        "filtered   : "
        + ", ".join(
            f"Top-{k}={overall_summary['family_filtered_xgboost'][f'top{k}']:.4f}"
            for k in [1, 3, 4, 5]
        )
    )
    print(
        "lift       : "
        + ", ".join(
            f"Top-{k}={overall_summary['lift'][f'top{k}']:+.4f}" for k in [1, 3, 4, 5]
        )
    )
    print("-" * 70)
    preview_columns = [
        "family",
        "sample_count",
        "candidate_species_count",
        "flat_top1",
        "filtered_top1",
        "top1_lift",
        "flat_top4",
        "filtered_top4",
        "top4_lift",
    ]
    print(family_summary_df[preview_columns].head(12).to_string(index=False))
    print("-" * 70)
    print(f"json_report : {output_dir / f'family_filtered_xgboost_report_{args.split}.json'}")
    print(f"csv_report  : {output_dir / f'family_breakdown_{args.split}.csv'}")


if __name__ == "__main__":
    main()
