import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import os
import json


class FeatureEngineer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_feature_distributions(self, df, target_column="species", sample_size=5000):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]

        n_features = len(numeric_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes.flatten()

        df_sample = df.sample(min(sample_size, len(df)), random_state=42)

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx] if n_features > 1 else axes[0]
            sns.histplot(
                data=df_sample, x=col, hue=target_column, kde=True, ax=ax, alpha=0.6
            )
            ax.set_title(f"{col} Distribution")

        for idx in range(n_features, len(axes) if isinstance(axes, np.ndarray) else 1):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "feature_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_correlation_matrix(self, df):
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "correlation_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return corr_matrix

    def analyze_pca(self, X, n_components=0.95):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("PCA Explained Variance by Component")

        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-")
        ax2.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("PCA Cumulative Explained Variance")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "pca_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        pca_results = {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
        }

        return pca_results, X_pca, pca

    def lasso_feature_selection(self, X, y, alpha=0.01):
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X, y)

        selector = SelectFromModel(lasso, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_importance = dict(zip(X.columns, np.abs(lasso.coef_)))

        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": np.abs(lasso.coef_)}
        ).sort_values("importance", ascending=True)

        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.set_xlabel("Importance (|Coefficient|)")
        ax.set_title("LASSO Feature Importance")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "lasso_feature_importance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        lasso_results = {
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "alpha": float(lasso.alpha_),
            "feature_importance": feature_importance,
        }

        return lasso_results, selector

    def plot_class_distribution(self, df, target_column="species"):
        plt.figure(figsize=(12, 6))
        value_counts = df[target_column].value_counts()

        ax = value_counts.plot(kind="bar")
        plt.title(f"{target_column} Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "class_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def perform_feature_engineering(self, df, X_processed, y, target_column="species"):
        print("Starting feature engineering...")

        self.plot_class_distribution(df, target_column)
        print("[OK] Class distribution plot saved")

        self.plot_feature_distributions(df, target_column)
        print("[OK] Feature distributions plot saved")

        corr_matrix = self.plot_correlation_matrix(df)
        print("[OK] Correlation matrix plot saved")

        pca_results, X_pca, pca_model = self.analyze_pca(X_processed)
        print(f"[OK] PCA analysis complete. Components: {pca_results['n_components']}")

        lasso_results, selector = self.lasso_feature_selection(X_processed, y)
        print(
            f"[OK] LASSO feature selection complete. Selected: {lasso_results['n_selected']} features"
        )

        results = {
            "pca": pca_results,
            "lasso": lasso_results,
            "correlation_matrix": corr_matrix.to_dict(),
        }

        results_path = os.path.join(self.output_dir, "feature_engineering_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results, X_pca, pca_model, selector


def perform_feature_engineering(df_path, X_path=None, output_dir="visualizations"):
    from preprocessing import preprocess_data

    df = pd.read_csv(df_path, encoding="utf-8")
    X, y, preprocessor = preprocess_data(df_path, output_dir="saved_models")

    engineer = FeatureEngineer(output_dir)
    results, X_pca, pca_model, selector = engineer.perform_feature_engineering(df, X, y)

    print("\nFeature engineering complete!")
    print(f"Results saved to {output_dir}")

    return results, X_pca, pca_model, selector


if __name__ == "__main__":
    results, X_pca, pca_model, selector = perform_feature_engineering(
        "../data/final_stonefly_dataset.csv"
    )
