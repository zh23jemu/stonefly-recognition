"""
高混淆物种对诊断脚本。

功能：
1. 读取 family_filtered_xgboost_analysis 生成的报告
2. 自动提取高混淆物种对
3. 统计每个物种对的 source_dataset、month、habitat、geo_cell 重叠情况
4. 输出物种级 confusion score，供 synthetic 生成脚本做限流
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description="统计高混淆物种对的数据构成与地理重叠")
    parser.add_argument(
        "--report-path",
        default=str(
            BACKEND_DIR
            / "saved_models_augmented"
            / "family_filtered_analysis"
            / "family_filtered_xgboost_report_test.json"
        ),
        help="family_filtered_xgboost_analysis 生成的 JSON 报告路径",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="训练数据 CSV 路径；不传时优先从报告中读取",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认写入报告所在目录",
    )
    parser.add_argument(
        "--confusion-key",
        choices=["filtered_top1_confusions", "flat_top1_confusions"],
        default="filtered_top1_confusions",
        help="使用哪类混淆对，默认 filtered_top1_confusions",
    )
    parser.add_argument(
        "--min-confusion-count",
        type=int,
        default=10,
        help="最少混淆次数，默认 10",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100,
        help="最多输出多少个混淆物种对，默认 100",
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


def normalize_count_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("unknown").astype(str).value_counts().to_dict().items()
    }


def limited_count_dict(series: pd.Series, limit: int = 8) -> dict[str, int]:
    counts = (
        series.fillna("unknown")
        .astype(str)
        .value_counts()
        .head(limit)
    )
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def geo_overlap_summary(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    cells_a = set(df_a["geo_cell"].dropna().astype(str))
    cells_b = set(df_b["geo_cell"].dropna().astype(str))
    intersection = cells_a & cells_b
    union = cells_a | cells_b
    return {
        "species_a_geo_cells": len(cells_a),
        "species_b_geo_cells": len(cells_b),
        "shared_geo_cells": len(intersection),
        "geo_jaccard": round(len(intersection) / len(union), 4) if union else 0.0,
        "species_a_overlap_ratio": round(len(intersection) / len(cells_a), 4) if cells_a else 0.0,
        "species_b_overlap_ratio": round(len(intersection) / len(cells_b), 4) if cells_b else 0.0,
        "shared_geo_cells_preview": sorted(list(intersection))[:12],
    }


def build_pair_summary(
    family_name: str,
    true_species: str,
    pred_species: str,
    confusion_count: int,
    dataset: pd.DataFrame,
) -> dict:
    true_rows = dataset[dataset["species"] == true_species].copy()
    pred_rows = dataset[dataset["species"] == pred_species].copy()

    return {
        "family": family_name,
        "true_species": true_species,
        "pred_species": pred_species,
        "confusion_count": int(confusion_count),
        "true_species_rows": int(len(true_rows)),
        "pred_species_rows": int(len(pred_rows)),
        "true_species_sources": normalize_count_dict(true_rows["source_dataset"]),
        "pred_species_sources": normalize_count_dict(pred_rows["source_dataset"]),
        "true_species_months": limited_count_dict(true_rows["month"]),
        "pred_species_months": limited_count_dict(pred_rows["month"]),
        "true_species_habitats": limited_count_dict(true_rows["habitat"]),
        "pred_species_habitats": limited_count_dict(pred_rows["habitat"]),
        "geo_overlap": geo_overlap_summary(true_rows, pred_rows),
    }


def build_species_scores(pair_summaries: list[dict]) -> list[dict]:
    species_scores: dict[str, dict] = {}

    for item in pair_summaries:
        family_name = item["family"]
        confusion_count = item["confusion_count"]

        for role in ["true_species", "pred_species"]:
            species_name = item[role]
            species_entry = species_scores.setdefault(
                species_name,
                {
                    "species": species_name,
                    "family": family_name,
                    "confusion_score": 0,
                    "true_confusion_score": 0,
                    "pred_confusion_score": 0,
                    "pair_count": 0,
                    "confusion_partners": set(),
                },
            )
            species_entry["confusion_score"] += confusion_count
            species_entry["pair_count"] += 1
            if role == "true_species":
                species_entry["true_confusion_score"] += confusion_count
                species_entry["confusion_partners"].add(item["pred_species"])
            else:
                species_entry["pred_confusion_score"] += confusion_count
                species_entry["confusion_partners"].add(item["true_species"])

    rows = []
    for species_name, item in species_scores.items():
        rows.append(
            {
                "species": species_name,
                "family": item["family"],
                "confusion_score": int(item["confusion_score"]),
                "true_confusion_score": int(item["true_confusion_score"]),
                "pred_confusion_score": int(item["pred_confusion_score"]),
                "pair_count": int(item["pair_count"]),
                "partner_count": int(len(item["confusion_partners"])),
                "confusion_partners": sorted(item["confusion_partners"]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["confusion_score"],
            row["true_confusion_score"],
            row["partner_count"],
            row["species"],
        ),
        reverse=True,
    )


def main():
    args = parse_args()
    report_path = resolve_path(args.report_path, BACKEND_DIR)
    if report_path is None or not report_path.exists():
        raise FileNotFoundError(f"分析报告不存在: {args.report_path}")

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    data_path = (
        resolve_path(args.data_path, REPO_ROOT)
        if args.data_path
        else resolve_path(report.get("data_path"), REPO_ROOT)
    )
    if data_path is None or not data_path.exists():
        raise FileNotFoundError("无法定位训练数据 CSV，请显式传入 --data-path")

    output_dir = (
        resolve_path(args.output_dir, BACKEND_DIR)
        if args.output_dir
        else report_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(data_path, low_memory=False)
    for column in ["source_dataset", "month", "habitat", "geo_cell"]:
        if column not in dataset.columns:
            dataset[column] = "unknown"

    pair_summaries = []
    for family_row in report.get("per_family", []):
        for pair in family_row.get(args.confusion_key, []):
            if int(pair["count"]) < args.min_confusion_count:
                continue
            pair_summaries.append(
                build_pair_summary(
                    family_name=family_row["family"],
                    true_species=pair["true_species"],
                    pred_species=pair["pred_species"],
                    confusion_count=int(pair["count"]),
                    dataset=dataset,
                )
            )

    pair_summaries = sorted(
        pair_summaries,
        key=lambda item: (item["confusion_count"], item["family"], item["true_species"]),
        reverse=True,
    )[: args.max_pairs]
    species_scores = build_species_scores(pair_summaries)

    pair_df = pd.DataFrame(
        [
            {
                "family": item["family"],
                "true_species": item["true_species"],
                "pred_species": item["pred_species"],
                "confusion_count": item["confusion_count"],
                "shared_geo_cells": item["geo_overlap"]["shared_geo_cells"],
                "geo_jaccard": item["geo_overlap"]["geo_jaccard"],
                "true_rows": item["true_species_rows"],
                "pred_rows": item["pred_species_rows"],
            }
            for item in pair_summaries
        ]
    )
    if not pair_df.empty:
        pair_df.to_csv(
            output_dir / "confusion_pair_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    species_df = pd.DataFrame(species_scores)
    if not species_df.empty:
        species_df.to_csv(
            output_dir / "species_confusion_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with open(output_dir / "confusion_pair_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(pair_summaries, file, ensure_ascii=False, indent=2)
    with open(output_dir / "species_confusion_scores.json", "w", encoding="utf-8") as file:
        json.dump(species_scores, file, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Confusion Pair Diagnostics")
    print("=" * 72)
    print(f"report_path : {report_path}")
    print(f"data_path   : {data_path}")
    print(f"pair_count  : {len(pair_summaries)}")
    print(f"species_cnt : {len(species_scores)}")
    if not pair_df.empty:
        print("-" * 72)
        print(pair_df.head(12).to_string(index=False))
    print("-" * 72)
    print(f"pairs_json  : {output_dir / 'confusion_pair_diagnostics.json'}")
    print(f"pairs_csv   : {output_dir / 'confusion_pair_diagnostics.csv'}")
    print(f"score_json  : {output_dir / 'species_confusion_scores.json'}")
    print(f"score_csv   : {output_dir / 'species_confusion_scores.csv'}")


if __name__ == "__main__":
    main()
