import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import warnings

warnings.filterwarnings("ignore")


class ModelEvaluator:
    def __init__(self, models_dir="saved_models", output_dir="visualizations"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluation_results = {}
        self.models = {}
        self.preprocessor = None
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)

    def load_models(self):
        model_names = ["random_forest", "svm", "xgboost", "knn"]
        for name in model_names:
            model_path = os.path.join(self.models_dir, f"{name}_model.pkl")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"[OK] Loaded {name} model")
        return self.models

    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division="warn"
        )
        recall = recall_score(y_test, y_pred, average="weighted", zero_division="warn")
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division="warn")
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division="warn")

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division="warn"
        )
        cm = confusion_matrix(y_test, y_pred)

        top_3_accuracy = None
        if y_pred_proba is not None:
            top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:]
            top_3_accuracy = float(
                np.mean([y_test[i] in top_3[i] for i in range(len(y_test))])
            )

        results = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "macro_f1": float(macro_f1),
            "top_3_accuracy": top_3_accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        print(f"\n{model_name.upper()} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Macro F1:  {macro_f1:.4f}")
        if top_3_accuracy is not None:
            print(f"  Top-3 Accuracy:  {top_3_accuracy:.4f}")

        return results, y_pred, y_pred_proba

    def evaluate_all_models(self, X_test, y_test):
        self.load_models()

        print("\n" + "=" * 50)
        print("Evaluating all models...")
        print("=" * 50)

        for name, model in self.models.items():
            results, y_pred, y_pred_proba = self.evaluate_model(
                model, X_test, y_test, name
            )
            self.evaluation_results[name] = results

            self.plot_confusion_matrix(
                results["confusion_matrix"], name, f"confusion_matrix_{name}.png"
            )

        return self.evaluation_results

    def plot_confusion_matrix(self, cm, model_name, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name.upper()}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  [OK] Saved confusion matrix for {model_name}")

    def compare_models(self):
        if not self.evaluation_results:
            print("No evaluation results available")
            return None

        comparison = {}
        for name, results in self.evaluation_results.items():
            comparison[name] = {
                "accuracy": results["accuracy"],
                "balanced_accuracy": results["balanced_accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1_score"],
                "macro_f1": results["macro_f1"],
                "top_3_accuracy": results["top_3_accuracy"],
            }

        comparison_df = pd.DataFrame(comparison).T

        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df.plot(kind="bar", ax=ax)
        plt.title("Model Performance Comparison")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend(loc="lower right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "model_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("\n" + "=" * 50)
        print("Model Comparison:")
        print("=" * 50)
        print(comparison_df.to_string())

        return comparison_df

    def select_best_model(self, metric="macro_f1"):
        if not self.evaluation_results:
            print("No evaluation results available")
            return None

        best_model_name = max(
            self.evaluation_results.items(), key=lambda x: x[1][metric]
        )[0]

        best_model = self.models[best_model_name]
        best_score = self.evaluation_results[best_model_name][metric]

        best_model_path = os.path.join(self.models_dir, "best_stonefly_model.pkl")
        joblib.dump(best_model, best_model_path)

        print(f"\n{'=' * 50}")
        print(f"Best Model: {best_model_name.upper()}")
        print(f"{metric}: {best_score:.4f}")
        print(f"{'=' * 50}")

        summary = {
            "best_model": best_model_name,
            "best_score": float(best_score),
            "metric_used": metric,
            "all_results": self.evaluation_results,
        }
        split_meta_path = os.path.join(self.models_dir, "dataset_split_metadata.json")
        if os.path.exists(split_meta_path):
            with open(split_meta_path, "r", encoding="utf-8") as f:
                summary["dataset"] = json.load(f)

        summary_path = os.path.join(self.models_dir, "model_evaluation_report.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return best_model_name, best_model


def evaluate_and_select_best_model(
    test_data_path="saved_models/test_data.npz",
    models_dir="saved_models",
    output_dir="visualizations",
):
    data = np.load(test_data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]

    evaluator = ModelEvaluator(models_dir, output_dir)
    evaluator.evaluate_all_models(X_test, y_test)
    evaluator.compare_models()
    best_name, best_model = evaluator.select_best_model()

    print(f"\nEvaluation complete! Best model: {best_name}")

    return evaluator, best_name, best_model


if __name__ == "__main__":
    evaluator, best_name, best_model = evaluate_and_select_best_model()
