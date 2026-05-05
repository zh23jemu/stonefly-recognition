import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


UNKNOWN_CATEGORY = "__UNKNOWN__"
UNKNOWN_SPECIES = "Unknown Stonefly"
MIN_SPECIES_COUNT = 50
FULL_FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "body_length_mm",
    "color",
    "head_feature",
]
HIERARCHICAL_FEATURES = [
    "lat",
    "lon",
    "country",
    "body_length_mm",
    "color",
    "head_feature",
]
NUMERIC_FEATURES = ["lat", "lon", "body_length_mm"]
CATEGORICAL_FEATURES = ["country", "color", "head_feature"]


class HierarchicalFeaturePreprocessor:
    """层级模型专用预处理器。

    这里故意不复用现有直接物种模型的 DataPreprocessor，因为层级模型不能把
    family 当作输入特征。该预处理器只学习 family 之外的 6 个特征，并在预测时
    对未见过的分类值统一映射到 __UNKNOWN__，避免线上输入中断。
    """

    def __init__(self):
        self.feature_columns = HIERARCHICAL_FEATURES.copy()
        self.numeric_features = NUMERIC_FEATURES.copy()
        self.categorical_features = CATEGORICAL_FEATURES.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.missing_values = {}

    def fit(self, df):
        fit_df = df[self.feature_columns].copy()

        for col in self.numeric_features:
            self.missing_values[col] = float(fit_df[col].median())
            fit_df[col] = fit_df[col].fillna(self.missing_values[col])

        for col in self.categorical_features:
            mode_value = fit_df[col].mode()[0] if not fit_df[col].mode().empty else UNKNOWN_CATEGORY
            self.missing_values[col] = str(mode_value)
            fit_df[col] = fit_df[col].fillna(mode_value).astype(str)
            encoder = LabelEncoder()
            encoder.fit(pd.Series(fit_df[col].unique().tolist() + [UNKNOWN_CATEGORY]))
            self.label_encoders[col] = encoder

        self.scaler.fit(fit_df[self.numeric_features])
        return self

    def transform(self, df):
        missing_columns = [col for col in self.feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要字段: {', '.join(missing_columns)}")

        transformed = df[self.feature_columns].copy()
        for col in self.numeric_features:
            transformed[col] = pd.to_numeric(transformed[col], errors="coerce")
            transformed[col] = transformed[col].fillna(self.missing_values[col])

        for col in self.categorical_features:
            encoder = self.label_encoders[col]
            known_values = set(encoder.classes_)
            transformed[col] = transformed[col].fillna(self.missing_values[col]).astype(str)
            transformed[col] = transformed[col].where(
                transformed[col].isin(known_values), UNKNOWN_CATEGORY
            )
            transformed[col] = encoder.transform(transformed[col])

        transformed[self.numeric_features] = self.scaler.transform(
            transformed[self.numeric_features]
        )
        return transformed

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def _prepare_known_species_dataset(data_path):
    """读取数据，并删除 Unknown 与低样本物种。

    层级模型和旧的直接 species 模型使用同一套数据过滤口径，保证两类实验的
    指标可比较：Unknown Stonefly 不参与训练，少于 50 条样本的物种也不参与。
    """
    df = pd.read_csv(data_path, encoding="utf-8")
    counts = df["species"].value_counts()
    valid_species = counts[
        (counts >= MIN_SPECIES_COUNT) & (counts.index != UNKNOWN_SPECIES)
    ].index
    filtered = df[
        df["species"].isin(valid_species) & (df["species"] != UNKNOWN_SPECIES)
    ].copy()
    return filtered.reset_index(drop=True), valid_species


def _split_dataset(df, random_state):
    y = df["species"].to_numpy()
    holdout_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state
    )
    train_idx, holdout_idx = next(
        holdout_splitter.split(np.arange(len(df)), y)
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    validation_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )
    test_idx, validation_idx = next(
        validation_splitter.split(
            np.arange(len(df_holdout)), df_holdout["species"].to_numpy()
        )
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    df_validation = df_holdout.iloc[validation_idx].reset_index(drop=True)
    return df_train, df_test, df_validation


def _sample_training_frame(df_train, max_train_samples, random_state):
    if not max_train_samples or len(df_train) <= max_train_samples:
        return df_train.reset_index(drop=True)

    sampler = StratifiedShuffleSplit(
        n_splits=1, train_size=max_train_samples, random_state=random_state
    )
    sampled_idx, _ = next(
        sampler.split(np.arange(len(df_train)), df_train["species"].to_numpy())
    )
    return df_train.iloc[sampled_idx].reset_index(drop=True)


def _build_xgboost(num_classes, random_state, n_jobs):
    """创建多分类 XGBoost 模型。

    参数控制得偏保守，目的是让 Slurm 上的正式层级训练能稳定完成，而不是再进入
    超长网格搜索。后续如果层级方案有效，再单独做小范围调参实验。
    """
    params = {
        "n_estimators": 350,
        "max_depth": 8,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "tree_method": "hist",
    }
    if num_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes
    return XGBClassifier(**params)


def _top_k_hit(y_true, proba, class_labels, k):
    order = np.argsort(proba, axis=1)[:, -k:]
    top_labels = class_labels[order]
    return float(np.mean([y_true[i] in top_labels[i] for i in range(len(y_true))]))


def _predict_species_with_family(model_bundle, X_row, family_name, fallback_species, limit=4):
    family_bundle = model_bundle["species_models"].get(family_name)
    if not family_bundle:
        return {
            "prediction": fallback_species,
            "confidence": 1.0,
            "top_predictions": [{"species": fallback_species, "probability": 1.0}],
            "fallback": True,
            "fallback_reason": "family_has_no_species_model",
        }

    model = family_bundle["model"]
    encoder = family_bundle["species_encoder"]
    proba = model.predict_proba(X_row)[0]
    class_labels = getattr(model, "classes_", np.arange(len(proba)))
    order = np.argsort(proba)[::-1][:limit]
    top_predictions = [
        {
            "species": encoder.inverse_transform([int(class_labels[index])])[0],
            "probability": float(proba[index]),
        }
        for index in order
    ]
    return {
        "prediction": top_predictions[0]["species"],
        "confidence": top_predictions[0]["probability"],
        "top_predictions": top_predictions,
        "fallback": False,
        "fallback_reason": None,
    }


def _evaluate_hierarchical(model_bundle, X_test, df_test):
    family_encoder = model_bundle["family_encoder"]
    family_proba = model_bundle["family_model"].predict_proba(X_test)
    family_class_labels = getattr(
        model_bundle["family_model"], "classes_", np.arange(family_proba.shape[1])
    )
    family_pred_encoded = family_class_labels[np.argmax(family_proba, axis=1)]
    family_pred = family_encoder.inverse_transform(family_pred_encoded.astype(int))
    family_true = df_test["family"].to_numpy()
    family_true_known_mask = np.isin(family_true, family_encoder.classes_)
    if np.any(family_true_known_mask):
        y_family_true_known = family_encoder.transform(family_true[family_true_known_mask])
        y_family_pred_known = family_encoder.transform(family_pred[family_true_known_mask])
        family_balanced_accuracy = float(
            balanced_accuracy_score(y_family_true_known, y_family_pred_known)
        )
        family_macro_f1 = float(
            f1_score(
                y_family_true_known,
                y_family_pred_known,
                average="macro",
                zero_division=0,
            )
        )
    else:
        family_balanced_accuracy = 0.0
        family_macro_f1 = 0.0

    species_true = df_test["species"].to_numpy()
    species_pred = []
    species_top3_hits = []
    species_top4_hits = []
    species_top5_hits = []
    oracle_species_pred = []

    fallback_by_family = model_bundle["fallback_species_by_family"]
    global_fallback = model_bundle["global_fallback_species"]

    for index in range(len(df_test)):
        X_row = X_test.iloc[[index]]
        predicted_family = family_pred[index]
        true_family = family_true[index]

        fallback_species = fallback_by_family.get(predicted_family, global_fallback)
        result = _predict_species_with_family(
            model_bundle, X_row, predicted_family, fallback_species, limit=5
        )
        top_species = [item["species"] for item in result["top_predictions"]]
        species_pred.append(result["prediction"])
        species_top3_hits.append(species_true[index] in top_species[:3])
        species_top4_hits.append(species_true[index] in top_species[:4])
        species_top5_hits.append(species_true[index] in top_species[:5])

        oracle_fallback = fallback_by_family.get(true_family, global_fallback)
        oracle_result = _predict_species_with_family(
            model_bundle, X_row, true_family, oracle_fallback
        )
        oracle_species_pred.append(oracle_result["prediction"])

    species_pred = np.asarray(species_pred)
    oracle_species_pred = np.asarray(oracle_species_pred)

    return {
        "family_accuracy": float(accuracy_score(family_true, family_pred)),
        "family_balanced_accuracy": family_balanced_accuracy,
        "family_macro_f1": family_macro_f1,
        "hierarchical_species_accuracy": float(accuracy_score(species_true, species_pred)),
        "hierarchical_species_macro_f1": float(
            f1_score(species_true, species_pred, average="macro", zero_division=0)
        ),
        "hierarchical_species_top_3_accuracy": float(np.mean(species_top3_hits)),
        "hierarchical_species_top_4_accuracy": float(np.mean(species_top4_hits)),
        "hierarchical_species_top_5_accuracy": float(np.mean(species_top5_hits)),
        "oracle_family_species_accuracy": float(
            accuracy_score(species_true, oracle_species_pred)
        ),
    }


def train_hierarchical_models(
    data_path,
    output_dir="saved_models",
    max_train_samples=20000,
    random_state=42,
    n_jobs=2,
):
    """训练 family -> species 两阶段层级模型，并保存全部线上预测资产。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df, valid_species = _prepare_known_species_dataset(data_path)
    df_train, df_test, df_validation = _split_dataset(df, random_state)
    df_fit = _sample_training_frame(df_train, max_train_samples, random_state)

    print("\n【层级模型】准备训练数据...")
    print(f"  [OK] 层级模型样本数: {len(df)}")
    print(f"  [OK] 层级模型物种数: {len(valid_species)}")
    print(f"  [OK] 层级训练样本数: {len(df_fit)}")
    print(f"  [OK] 层级测试样本数: {len(df_test)}")

    preprocessor = HierarchicalFeaturePreprocessor()
    X_train = preprocessor.fit_transform(df_fit)
    X_test = preprocessor.transform(df_test)

    family_encoder = LabelEncoder()
    # XGBoost 要求训练标签是连续的 0..n-1，因此 family 编码器只基于实际参与
    # 训练的抽样数据拟合。测试集中未被训练到的稀有 family 不可能被预测出来，
    # 评估时直接使用 family 字符串比较即可。
    y_family = family_encoder.fit_transform(df_fit["family"])
    family_model = _build_xgboost(
        num_classes=len(family_encoder.classes_),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    print("  [层级模型] 开始训练 family XGBoost...")
    family_model.fit(X_train, y_family)
    print("  [OK] family XGBoost 训练完成")

    species_models = {}
    skipped_families = {}
    fallback_species_by_family = {
        family_name: family_df["species"].value_counts().idxmax()
        for family_name, family_df in df_train.groupby("family")
    }
    for family_name, family_df in df_fit.groupby("family"):
        species_counts = family_df["species"].value_counts()
        if family_df["species"].nunique() < 2:
            skipped_families[family_name] = "species_count_less_than_2"
            continue

        species_encoder = LabelEncoder()
        y_species = species_encoder.fit_transform(family_df["species"])
        X_family = preprocessor.transform(family_df)
        species_model = _build_xgboost(
            num_classes=len(species_encoder.classes_),
            random_state=random_state,
            n_jobs=n_jobs,
        )
        print(
            f"  [层级模型] 训练 {family_name} species 子模型: "
            f"{len(family_df)} 样本, {len(species_encoder.classes_)} 类"
        )
        species_model.fit(X_family, y_species)
        species_models[family_name] = {
            "model": species_model,
            "species_encoder": species_encoder,
            "classes": species_encoder.classes_.tolist(),
        }

    global_fallback_species = df_fit["species"].value_counts().idxmax()
    model_bundle = {
        "preprocessor": preprocessor,
        "family_model": family_model,
        "family_encoder": family_encoder,
        "species_models": species_models,
        "fallback_species_by_family": fallback_species_by_family,
        "global_fallback_species": global_fallback_species,
        "feature_columns": HIERARCHICAL_FEATURES,
        "full_feature_columns": FULL_FEATURES,
        "skipped_families": skipped_families,
    }

    print("  [层级模型] 开始评估...")
    metrics = _evaluate_hierarchical(model_bundle, X_test, df_test)
    print(f"  Family Accuracy: {metrics['family_accuracy']:.4f}")
    print(
        "  Hierarchical Species Accuracy: "
        f"{metrics['hierarchical_species_accuracy']:.4f}"
    )
    print(
        "  Hierarchical Species Top-4 Accuracy: "
        f"{metrics['hierarchical_species_top_4_accuracy']:.4f}"
    )
    print(
        "  Oracle-Family Species Accuracy: "
        f"{metrics['oracle_family_species_accuracy']:.4f}"
    )

    joblib.dump(family_model, output_path / "family_xgboost_model.pkl")
    joblib.dump(model_bundle, output_path / "hierarchical_model_bundle.pkl")

    species_dir = output_path / "species_models_by_family"
    species_dir.mkdir(parents=True, exist_ok=True)
    for family_name, family_bundle in species_models.items():
        safe_name = family_name.replace("/", "_").replace("\\", "_")
        joblib.dump(family_bundle, species_dir / f"{safe_name}_species_model.pkl")

    results = {
        "model_type": "hierarchical_family_then_species",
        "family_model": "xgboost",
        "species_model": "xgboost_by_family",
        "unknown_policy": "removed",
        "min_species_count": MIN_SPECIES_COUNT,
        "features_used": HIERARCHICAL_FEATURES,
        "features_excluded_as_labels": ["family", "species"],
        "dataset": {
            "samples": int(len(df)),
            "species_count": int(df["species"].nunique()),
            "family_count": int(df["family"].nunique()),
            "train_samples_used": int(len(df_fit)),
            "test_samples": int(len(df_test)),
            "validation_samples": int(len(df_validation)),
        },
        "metrics": metrics,
        "trained_species_families": sorted(species_models.keys()),
        "skipped_families": skipped_families,
    }
    with open(output_path / "hierarchical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
