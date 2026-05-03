import argparse
import csv
import json
import importlib
import os
import sys
from datetime import datetime

backend_root = os.path.dirname(os.path.dirname(__file__))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

run_complete_ml_pipeline = importlib.import_module("ml").run_complete_ml_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Unknown cap ratios")
    parser.add_argument(
        "--caps",
        nargs="+",
        type=float,
        default=[0.15, 0.2, 0.25],
        help="Unknown cap ratios to evaluate",
    )
    parser.add_argument("--cv", type=int, default=3, help="CV folds for sweep")
    parser.add_argument(
        "--data-path",
        default="../data/final_stonefly_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--base-output-dir",
        default="sweep_models",
        help="Base output directory for model artifacts",
    )
    parser.add_argument(
        "--base-viz-dir",
        default="sweep_visualizations",
        help="Base output directory for visualizations",
    )
    parser.add_argument(
        "--report-dir",
        default="visualizations",
        help="Directory to write comparison report",
    )
    return parser.parse_args()


def safe_tag(value):
    return str(value).replace(".", "p")


def run_sweep(args):
    os.makedirs(args.base_output_dir, exist_ok=True)
    os.makedirs(args.base_viz_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    rows = []
    for cap in args.caps:
        tag = safe_tag(cap)
        output_dir = os.path.join(args.base_output_dir, f"cap_{tag}")
        viz_dir = os.path.join(args.base_viz_dir, f"cap_{tag}")

        print("=" * 70)
        print(f"Running experiment: unknown_cap_ratio={cap}, cv={args.cv}")
        print("=" * 70)

        evaluator, best_name, _ = run_complete_ml_pipeline(
            data_path=args.data_path,
            output_dir=output_dir,
            viz_dir=viz_dir,
            cv=args.cv,
            unknown_cap_ratio=cap,
            random_state=42,
        )

        best_metrics = evaluator.evaluation_results[best_name]
        row = {
            "unknown_cap_ratio": cap,
            "best_model": best_name,
            "metric_used": "macro_f1",
            "macro_f1": float(best_metrics.get("macro_f1", 0.0)),
            "accuracy": float(best_metrics.get("accuracy", 0.0)),
            "balanced_accuracy": float(best_metrics.get("balanced_accuracy", 0.0)),
            "top_3_accuracy": float(best_metrics.get("top_3_accuracy", 0.0) or 0.0),
            "unknown_prediction_rate": float(
                best_metrics.get("unknown_prediction_rate", 0.0) or 0.0
            ),
            "known_only_accuracy": float(
                best_metrics.get("known_only_accuracy", 0.0) or 0.0
            ),
            "output_dir": output_dir,
            "viz_dir": viz_dir,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["macro_f1"], reverse=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.report_dir, f"unknown_cap_sweep_{timestamp}.json")
    csv_path = os.path.join(args.report_dir, f"unknown_cap_sweep_{timestamp}.csv")
    md_path = os.path.join(args.report_dir, f"unknown_cap_sweep_{timestamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Unknown 占比上限对比实验\n\n")
        f.write(
            "| cap | best_model | macro_f1 | accuracy | balanced_accuracy | top3 | unknown_rate | known_only |\n"
        )
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['unknown_cap_ratio']:.2f} | {row['best_model']} | {row['macro_f1']:.4f} | "
                f"{row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['top_3_accuracy']:.4f} | "
                f"{row['unknown_prediction_rate']:.4f} | {row['known_only_accuracy']:.4f} |\n"
            )

        best = rows[0]
        f.write("\n## 推荐参数\n\n")
        f.write(
            f"- 推荐 `unknown_cap_ratio={best['unknown_cap_ratio']:.2f}`，该配置在本次实验中 `macro_f1={best['macro_f1']:.4f}`。\n"
        )

    print("\nSweep finished.")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print(f"MD report:   {md_path}")


if __name__ == "__main__":
    run_sweep(parse_args())
