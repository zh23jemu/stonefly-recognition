import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ml import run_complete_ml_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train stonefly classification models")
    parser.add_argument(
        "--data-path",
        default="../data/final_stonefly_dataset.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument("--output-dir", default="saved_models", help="Model output dir")
    parser.add_argument(
        "--viz-dir", default="visualizations", help="Visualization output dir"
    )
    parser.add_argument("--cv", type=int, default=5, help="Cross validation folds")
    parser.add_argument(
        "--unknown-cap-ratio",
        type=float,
        default=0.2,
        help="Maximum Unknown class ratio after downsampling (0-1)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("开始训练石蝇分类模型...")
    print("=" * 60)
    print(
        f"参数: cv={args.cv}, unknown_cap_ratio={args.unknown_cap_ratio}, random_state={args.random_state}"
    )

    evaluator, best_name, best_model = run_complete_ml_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
        cv=args.cv,
        unknown_cap_ratio=args.unknown_cap_ratio,
        random_state=args.random_state,
    )

    print("\n训练完成！")
    print(f"最优模型: {best_name}")
    print("=" * 60)
