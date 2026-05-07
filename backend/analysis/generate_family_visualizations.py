"""
科级分类可视化生成脚本。

用途：
1. 基于 family_clean 数据集生成“类别分布 / 特征分布 / 相关矩阵”图
2. 基于 family_clean 评估报告生成“模型对比”图
3. 同步生成 data_exploration_report.json，供前端大屏读取

说明：
- 该脚本专门服务于当前“科级分类上线版”页面
- 输出目录默认写入 backend/visualizations_family_clean/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR.parent / "data" / "stonefly_combined_data_family_clean.csv"
REPORT_PATH = ROOT_DIR / "saved_models_family_clean" / "model_evaluation_report.json"
OUTPUT_DIR = ROOT_DIR / "visualizations_family_clean"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, low_memory=False)


def _save_data_exploration_report(df: pd.DataFrame) -> None:
    """生成前端大屏继续复用的数据探索摘要。"""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    family_counts = df["family"].value_counts()

    report = {
        "basic_statistics": {
            "total_samples": int(len(df)),
            "total_features": int(len(df.columns)),
            "feature_names": df.columns.tolist(),
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
        },
        "target_analysis": {
            "target_column": "family",
            "num_classes": int(df["family"].nunique()),
            "class_distribution": family_counts.to_dict(),
            "class_percentages": (family_counts / len(df) * 100).round(4).to_dict(),
            "imbalance_ratio": float(family_counts.max() / family_counts.min()),
        },
    }

    output_path = OUTPUT_DIR / "data_exploration_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_class_distribution(df: pd.DataFrame) -> None:
    """绘制科级样本分布图，替代旧的物种类别分布图。"""
    family_counts = df["family"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x=family_counts.index, y=family_counts.values, palette="Blues_r")
    ax.set_title("Family Class Distribution", fontsize=14)
    ax.set_xlabel("Family")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_feature_distributions(df: pd.DataFrame) -> None:
    """绘制与科级分类直接相关的核心特征分布。"""
    plot_columns = [
        ("body_length_mm", "Body Length (mm)"),
        ("lat", "Latitude"),
        ("lon", "Longitude"),
        ("month", "Month"),
        ("habitat", "Habitat"),
        ("color", "Color"),
    ]
    available_columns = [(col, title) for col, title in plot_columns if col in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    df_sample = df.sample(min(len(df), 12000), random_state=42).copy()
    top_families = df_sample["family"].value_counts().head(8).index
    df_sample = df_sample[df_sample["family"].isin(top_families)]

    for idx, (column, title) in enumerate(available_columns):
        ax = axes[idx]
        if pd.api.types.is_numeric_dtype(df_sample[column]):
            sns.histplot(
                data=df_sample,
                x=column,
                hue="family",
                kde=True,
                stat="density",
                common_norm=False,
                alpha=0.22,
                ax=ax,
                legend=idx == 0,
            )
        else:
            top_categories = df_sample[column].astype(str).value_counts().head(8).index
            filtered = df_sample[df_sample[column].astype(str).isin(top_categories)]
            sns.countplot(
                data=filtered,
                x=column,
                hue="family",
                order=top_categories,
                ax=ax,
            )
            ax.tick_params(axis="x", rotation=25)

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for idx in range(len(available_columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_correlation_matrix(df: pd.DataFrame) -> None:
    """绘制数值特征相关矩阵。"""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_model_comparison() -> None:
    """从 family_clean 评估报告绘制当前三模型对比图。"""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    evaluation_results = report.get("evaluation_results", {})
    if not evaluation_results:
        raise ValueError("model_evaluation_report.json 中缺少 evaluation_results")

    rows = []
    for model_name, metrics in evaluation_results.items():
        rows.append(
            {
                "model": model_name.upper(),
                "Top-1": metrics.get("top1_accuracy", 0.0),
                "Balanced": metrics.get("balanced_accuracy", 0.0),
                "F1-macro": metrics.get("f1_macro", 0.0),
                "Top-3": metrics.get("top3_accuracy", 0.0),
            }
        )

    comparison_df = pd.DataFrame(rows).set_index("model")
    ax = comparison_df.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), rot=0)
    ax.set_title("Family Classification Model Comparison", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    _ensure_output_dir()
    df = _load_data()
    _save_data_exploration_report(df)
    _plot_class_distribution(df)
    _plot_feature_distributions(df)
    _plot_correlation_matrix(df)
    _plot_model_comparison()
    print(f"family_clean 可视化已生成到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
