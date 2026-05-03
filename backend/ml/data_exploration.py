import pandas as pd
import numpy as np
import json
import os


def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    return df


def basic_statistics(df):
    stats = {
        "total_samples": int(len(df)),
        "total_features": int(len(df.columns)),
        "feature_names": df.columns.tolist(),
        "numeric_features": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": df.select_dtypes(include=["object"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    return stats


def target_analysis(df, target_column="species"):
    target_counts = df[target_column].value_counts()
    target_stats = {
        "num_classes": int(len(target_counts)),
        "class_distribution": target_counts.to_dict(),
        "class_percentages": (target_counts / len(df) * 100).to_dict(),
        "imbalance_ratio": float(target_counts.max() / target_counts.min()),
    }
    return target_stats


def numeric_features_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis = {}
    for col in numeric_cols:
        analysis[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }
    return analysis


def categorical_features_analysis(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    analysis = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        analysis[col] = {
            "unique_values": int(df[col].nunique()),
            "top_values": value_counts.head(10).to_dict(),
            "value_percentages": (value_counts / len(df) * 100).head(10).to_dict(),
        }
    return analysis


def detect_outliers(df, method="iqr", threshold=1.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        outliers[col] = {
            "count": int(outlier_count),
            "percentage": float(outlier_count / len(df) * 100),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }
    return outliers


def explore_data(file_path, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(file_path)

    report = {
        "basic_statistics": basic_statistics(df),
        "target_analysis": target_analysis(df),
        "numeric_analysis": numeric_features_analysis(df),
        "categorical_analysis": categorical_features_analysis(df),
        "outliers": detect_outliers(df),
    }

    report_path = os.path.join(output_dir, "data_exploration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report, df


if __name__ == "__main__":
    report, df = explore_data("../data/final_stonefly_dataset.csv")
    print(f"数据探索完成!")
    print(f"样本数: {report['basic_statistics']['total_samples']}")
    print(f"特征数: {report['basic_statistics']['total_features']}")
    print(f"类别数: {report['target_analysis']['num_classes']}")
