import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


UNKNOWN_SPECIES = "Unknown Stonefly"

OUTPUT_COLUMNS = [
    "species",
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
    "source_dataset",
    "gbifID",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build one combined CSV from data + data2")
    parser.add_argument(
        "--main-data",
        default="data/final_stonefly_dataset.csv",
        help="当前主数据集 CSV 路径",
    )
    parser.add_argument(
        "--data2-occurrence",
        default="data2/occurrence.txt",
        help="data2 中的 occurrence.txt 路径",
    )
    parser.add_argument(
        "--output",
        default="data/stonefly_combined_data_data2.csv",
        help="输出合并 CSV 路径",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="保留 Unknown Stonefly；默认会删除 Unknown Stonefly",
    )
    return parser.parse_args()


def normalize_species_name(value):
    """把带作者信息的学名统一成“属名 种加词”。

    data2 的 scientificName 通常类似 `Allocapnia mohri Ross & Ricker, 1964`，
    当前主数据集使用 `Allocapnia mohri`，所以这里统一成前两个单词。
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = value.strip().split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def normalize_habitat(value):
    """把 habitat 文本归并成少量稳定类别，减少拼写差异。"""
    text = str(value).strip().lower()
    if not text or text in {"nan", "unknown", "none", "null"}:
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


def add_common_fields(df):
    """补齐合并 CSV 的公共派生字段。"""
    result = df.copy()
    result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
    result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
    result["body_length_mm"] = pd.to_numeric(result["body_length_mm"], errors="coerce")
    result["country"] = result["country"].fillna("unknown").astype(str).str.upper()
    result["family"] = result["family"].fillna("unknown").astype(str)
    result["month"] = result["month"].fillna("unknown")
    result["season"] = result["month"].map(month_to_season)
    result["habitat"] = result["habitat"].fillna("unknown").astype(str).str.lower()
    result["habitat_group"] = result["habitat"].map(normalize_habitat)
    result["sex"] = result["sex"].fillna("unknown").astype(str).str.lower()
    result["life_stage"] = result["life_stage"].fillna("unknown").astype(str).str.lower()
    result["color"] = result["color"].fillna("unknown").astype(str).str.lower()
    result["head_feature"] = result["head_feature"].fillna("unknown").astype(str).str.lower()
    result["gbifID"] = result["gbifID"].fillna("").astype(str)

    result["lat_bin"] = result["lat"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    result["lon_bin"] = result["lon"].map(lambda x: math.floor(x) if pd.notna(x) else np.nan)
    result["geo_cell"] = (
        result["lat_bin"].fillna("unknown").astype(str)
        + "_"
        + result["lon_bin"].fillna("unknown").astype(str)
    )
    return result


def load_main_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source_dataset"] = "data"
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
            "body_length_mm": np.nan,
            "color": "unknown",
            "head_feature": "unknown",
            "month": occurrence.get("month", "unknown"),
            "habitat": occurrence.get("habitat", "unknown"),
            "sex": occurrence.get("sex", "unknown"),
            "life_stage": occurrence.get("lifeStage", "unknown"),
            "source_dataset": "data2",
            "gbifID": occurrence.get("gbifID", ""),
        }
    )
    df = df.replace({"": np.nan})
    df = df.dropna(subset=["species", "lat", "lon", "family"])
    return df.reset_index(drop=True)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_df = load_main_dataset(args.main_data)
    data2_df = load_data2_dataset(args.data2_occurrence)
    combined = pd.concat([main_df, data2_df], ignore_index=True)
    combined = add_common_fields(combined)

    combined = combined.dropna(subset=["species", "lat", "lon", "family"])
    if not args.keep_unknown:
        combined = combined[combined["species"] != UNKNOWN_SPECIES]

    combined = combined[OUTPUT_COLUMNS].reset_index(drop=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "output": str(output_path),
        "rows": int(len(combined)),
        "species_count": int(combined["species"].nunique()),
        "family_count": int(combined["family"].nunique()),
        "source_counts": combined["source_dataset"].value_counts().to_dict(),
        "columns": OUTPUT_COLUMNS,
    }
    print(summary)


if __name__ == "__main__":
    main()
