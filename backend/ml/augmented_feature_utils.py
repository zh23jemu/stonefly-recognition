import math

import numpy as np
import pandas as pd


# 增强数据集的基础输入特征。前 11 个是页面可直接输入或已有训练脚本使用的字段，
# 后 5 个是从原始输入自动派生出的结构化辅助特征。
BASE_AUGMENTED_FEATURES = [
    "lat",
    "lon",
    "country",
    "family",
    "body_length_mm",
    "color",
    "head_feature",
    "month",
    "habitat",
    "sex",
    "life_stage",
]
OPTIONAL_DERIVED_FEATURES = [
    "season",
    "habitat_group",
    "lat_bin",
    "lon_bin",
    "geo_cell",
]
ALL_AUGMENTED_MODEL_FEATURES = BASE_AUGMENTED_FEATURES + OPTIONAL_DERIVED_FEATURES
HIERARCHICAL_BASE_FEATURES = [col for col in BASE_AUGMENTED_FEATURES if col != "family"]
HIERARCHICAL_OPTIONAL_FEATURES = OPTIONAL_DERIVED_FEATURES.copy()
ALL_HIERARCHICAL_MODEL_FEATURES = (
    HIERARCHICAL_BASE_FEATURES + HIERARCHICAL_OPTIONAL_FEATURES
)

NUMERIC_FEATURE_CANDIDATES = {
    "lat",
    "lon",
    "body_length_mm",
    "lat_bin",
    "lon_bin",
}
STRING_FEATURE_CANDIDATES = {
    "country",
    "family",
    "color",
    "head_feature",
    "month",
    "season",
    "habitat",
    "habitat_group",
    "sex",
    "life_stage",
    "geo_cell",
}
MODEL_EXCLUDED_COLUMNS = {"species", "source_dataset", "gbifID"}


def _is_missing(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _normalize_string(value, default="unknown") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def normalize_month(value) -> str:
    """统一 month 字段，确保训练和预测都使用同一套离散月份表示。"""
    if _is_missing(value):
        return "unknown"

    text = str(value).strip()
    lowered = text.lower()
    if lowered == "unknown":
        return "unknown"

    try:
        number = float(text)
    except ValueError:
        return text

    if not np.isfinite(number):
        return "unknown"

    month = int(number)
    if 1 <= month <= 12:
        return str(month)
    return "unknown"


def derive_season(month_value) -> str:
    """按照增强数据集的四季口径从 month 推导 season。"""
    normalized_month = normalize_month(month_value)
    if normalized_month == "unknown":
        return "unknown"

    month = int(normalized_month)
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    if month in {9, 10, 11}:
        return "fall"
    return "winter"


def derive_habitat_group(habitat_value) -> str:
    """把细粒度 habitat 归并到增强数据集中的 habitat_group。"""
    habitat = _normalize_string(habitat_value).lower()
    if habitat == "unknown":
        return "unknown"
    if habitat == "lake":
        return "lake"
    if "pond" in habitat or "slough" in habitat:
        return "lake_pond"
    if "waterfall" in habitat:
        return "waterfall"
    if "spring" in habitat or "seep" in habitat or "well" in habitat:
        return "spring"
    if "river" in habitat:
        return "river"
    if "stream" in habitat:
        return "stream"
    return "other"


def _derive_spatial_bins(df: pd.DataFrame) -> pd.DataFrame:
    lat_numeric = pd.to_numeric(df.get("lat"), errors="coerce")
    lon_numeric = pd.to_numeric(df.get("lon"), errors="coerce")

    if "lat_bin" not in df.columns:
        df["lat_bin"] = np.nan
    if "lon_bin" not in df.columns:
        df["lon_bin"] = np.nan

    lat_bin_existing = pd.to_numeric(df["lat_bin"], errors="coerce")
    lon_bin_existing = pd.to_numeric(df["lon_bin"], errors="coerce")

    lat_bin_derived = lat_numeric.map(
        lambda value: math.floor(value) if pd.notna(value) else np.nan
    )
    lon_bin_derived = lon_numeric.map(
        lambda value: math.floor(value) if pd.notna(value) else np.nan
    )

    df["lat_bin"] = lat_bin_existing.where(lat_bin_existing.notna(), lat_bin_derived)
    df["lon_bin"] = lon_bin_existing.where(lon_bin_existing.notna(), lon_bin_derived)

    if "geo_cell" not in df.columns:
        df["geo_cell"] = "unknown"

    geo_cell_series = df["geo_cell"].map(_normalize_string)
    missing_geo_mask = geo_cell_series == "unknown"
    lat_bin_int = df["lat_bin"].dropna().astype(int)
    lon_bin_int = df["lon_bin"].dropna().astype(int)
    geo_cell_derived = pd.Series("unknown", index=df.index, dtype=object)
    valid_geo_mask = df["lat_bin"].notna() & df["lon_bin"].notna()
    geo_cell_derived.loc[valid_geo_mask] = (
        df.loc[valid_geo_mask, "lat_bin"].astype(int).astype(str)
        + "_"
        + df.loc[valid_geo_mask, "lon_bin"].astype(int).astype(str)
    )
    df["geo_cell"] = geo_cell_series.where(~missing_geo_mask, geo_cell_derived)
    return df


def prepare_augmented_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """标准化增强数据集字段，并在缺失时补齐可派生特征。"""
    normalized = df.copy()

    for column in ["lat", "lon", "body_length_mm"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "month" in normalized.columns:
        normalized["month"] = normalized["month"].map(normalize_month)
    else:
        normalized["month"] = "unknown"

    if "season" in normalized.columns:
        normalized["season"] = normalized["season"].map(_normalize_string)
        normalized["season"] = normalized["season"].where(
            normalized["season"] != "unknown",
            normalized["month"].map(derive_season),
        )
    else:
        normalized["season"] = normalized["month"].map(derive_season)

    if "habitat" in normalized.columns:
        normalized["habitat"] = normalized["habitat"].map(_normalize_string)
    else:
        normalized["habitat"] = "unknown"

    if "habitat_group" in normalized.columns:
        normalized["habitat_group"] = normalized["habitat_group"].map(_normalize_string)
        normalized["habitat_group"] = normalized["habitat_group"].where(
            normalized["habitat_group"] != "unknown",
            normalized["habitat"].map(derive_habitat_group),
        )
    else:
        normalized["habitat_group"] = normalized["habitat"].map(derive_habitat_group)

    for column in [
        "country",
        "family",
        "color",
        "head_feature",
        "sex",
        "life_stage",
        "source_dataset",
    ]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_normalize_string)

    normalized = _derive_spatial_bins(normalized)

    if "geo_cell" in normalized.columns:
        normalized["geo_cell"] = normalized["geo_cell"].map(_normalize_string)

    return normalized


def get_flat_model_feature_columns() -> list[str]:
    """增强版平铺模型使用的特征顺序。"""
    return ALL_AUGMENTED_MODEL_FEATURES.copy()


def get_hierarchical_feature_columns() -> list[str]:
    """增强版层级模型使用的特征顺序。"""
    return ALL_HIERARCHICAL_MODEL_FEATURES.copy()


def build_feature_frame_from_payload(data: dict, feature_columns: list[str]) -> pd.DataFrame:
    """把接口输入转换成训练同口径的一行特征 DataFrame。"""
    raw_df = pd.DataFrame([data])
    prepared_df = prepare_augmented_dataframe(raw_df)

    for column in feature_columns:
        if column not in prepared_df.columns:
            if column in NUMERIC_FEATURE_CANDIDATES:
                prepared_df[column] = np.nan
            else:
                prepared_df[column] = "unknown"

    return prepared_df[feature_columns].copy()
