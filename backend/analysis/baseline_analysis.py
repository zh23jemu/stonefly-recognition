import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    models_dir = os.path.join(base_dir, "saved_models")
    output_path = os.path.join(base_dir, "visualizations", "baseline_metrics.json")

    model = joblib.load(os.path.join(models_dir, "best_stonefly_model.pkl"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    test_data = np.load(os.path.join(models_dir, "test_data.npz"))

    x_test = test_data["X_test"]
    y_test = test_data["y_test"]

    y_pred = model.predict(x_test)

    true_labels = preprocessor.target_encoder.inverse_transform(y_test)
    pred_labels = preprocessor.target_encoder.inverse_transform(y_pred)

    unknown_label = "Unknown Stonefly"
    unknown_pred_mask = pred_labels == unknown_label
    unknown_true_mask = true_labels == unknown_label
    known_true_mask = ~unknown_true_mask

    overall_accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0.0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0.0))

    unknown_prediction_rate = float(np.mean(unknown_pred_mask))
    unknown_true_rate = float(np.mean(unknown_true_mask))

    if np.any(known_true_mask):
        known_only_accuracy = float(
            np.mean(pred_labels[known_true_mask] == true_labels[known_true_mask])
        )
    else:
        known_only_accuracy = 0.0

    top_pred_classes, top_pred_counts = np.unique(pred_labels, return_counts=True)
    order = np.argsort(top_pred_counts)[::-1]
    top_10_predictions = [
        {
            "species": str(top_pred_classes[i]),
            "count": int(top_pred_counts[i]),
            "rate": float(top_pred_counts[i] / len(pred_labels)),
        }
        for i in order[:10]
    ]

    metrics = {
        "test_samples": int(len(y_test)),
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "unknown_prediction_rate": unknown_prediction_rate,
        "unknown_true_rate": unknown_true_rate,
        "known_only_accuracy": known_only_accuracy,
        "top_10_predicted_classes": top_10_predictions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Baseline Metrics")
    print("=" * 60)
    print(f"test_samples: {metrics['test_samples']}")
    print(f"overall_accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print(f"unknown_prediction_rate: {metrics['unknown_prediction_rate']:.4f}")
    print(f"unknown_true_rate: {metrics['unknown_true_rate']:.4f}")
    print(f"known_only_accuracy: {metrics['known_only_accuracy']:.4f}")
    print(f"saved_report: {output_path}")


if __name__ == "__main__":
    main()
