import argparse
import json
from datetime import datetime
from pathlib import Path


DEFAULT_STATE_FILE = (
    Path(__file__).resolve().parents[1]
    / "saved_models"
    / "model_experiments"
    / "catboost_lightgbm_ensemble_state.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Show CatBoost/LightGBM experiment progress")
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="Path to the persisted experiment state JSON",
    )
    return parser.parse_args()


def format_elapsed(started_at):
    if not started_at:
        return "未知"
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return "未知"
    seconds = int((datetime.now() - started).total_seconds())
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes}分钟{sec}秒"


def main():
    args = parse_args()
    state_path = Path(args.state_file)
    if not state_path.exists():
        print(f"未找到进度文件: {state_path}")
        print("说明: 实验脚本启动后会先创建该文件。")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    completed = state.get("completed", [])
    current_model = state.get("current_model")
    current_started = state.get("current_model_started_at")

    print(f"进度文件: {state_path}")
    print(f"实验开始时间: {state.get('started_at')}")
    print(f"最后更新时间: {state.get('last_update_at')}")
    print(f"已完成模型: {', '.join(completed) if completed else '暂无'}")
    if current_model:
        print(f"当前模型: {current_model}")
        print(f"当前模型已运行: {format_elapsed(current_started)}")
    else:
        print("当前模型: 暂无运行中模型")

    results = state.get("results", [])
    if results:
        print("\n已有结果:")
        for row in results:
            print(
                f"- {row['model']}: accuracy={row.get('accuracy', 0):.4f}, "
                f"macro_f1={row.get('macro_f1', 0):.4f}, "
                f"top4={row.get('top_4_accuracy', 0):.4f}"
            )


if __name__ == "__main__":
    main()
