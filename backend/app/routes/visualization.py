from flask import Blueprint, jsonify, send_file
import os
import json

visualization_bp = Blueprint("visualization", __name__)

VISUALIZATION_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "visualizations_family_clean"),
    os.path.join(os.path.dirname(__file__), "..", "..", "visualizations_family"),
    os.path.join(os.path.dirname(__file__), "..", "..", "visualizations_augmented"),
    os.path.join(os.path.dirname(__file__), "..", "..", "visualizations"),
]
MODELS_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "saved_models_family_clean"),
    os.path.join(os.path.dirname(__file__), "..", "..", "saved_models_family"),
    os.path.join(os.path.dirname(__file__), "..", "..", "saved_models_augmented"),
    os.path.join(os.path.dirname(__file__), "..", "..", "saved_models"),
]


def _resolve_report_path() -> str | None:
    """优先读取最新上线目录中的模型评估报告。"""
    for model_dir in MODELS_DIR_CANDIDATES:
        report_path = os.path.join(model_dir, "model_evaluation_report.json")
        if os.path.exists(report_path):
            return report_path
    return None


def _resolve_visualization_dir() -> str:
    """优先读取当前 family_clean 版本的可视化目录。"""
    for directory in VISUALIZATION_DIR_CANDIDATES:
        if os.path.exists(directory):
            return directory
    return VISUALIZATION_DIR_CANDIDATES[-1]


def _build_model_info_payload(report: dict) -> dict:
    """把训练报告转换成前端大屏一直在消费的统一结构。"""
    evaluation_results = report.get("evaluation_results", {})
    meta = report.get("meta", {})

    all_results = {}
    best_model = ""
    best_score = -1.0

    for model_name, metrics in evaluation_results.items():
        top1_accuracy = metrics.get("top1_accuracy", 0.0)
        if top1_accuracy > best_score:
            best_model = model_name
            best_score = top1_accuracy

        all_results[model_name] = {
            "accuracy": top1_accuracy,
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "precision": metrics.get("f1_weighted", 0.0),
            "recall": metrics.get("top3_accuracy", 0.0),
            "f1_score": metrics.get("f1_weighted", 0.0),
            "macro_f1": metrics.get("f1_macro", 0.0),
            "top_3_accuracy": metrics.get("top3_accuracy", 0.0),
            "top_4_accuracy": metrics.get("top4_accuracy", metrics.get("top3_accuracy", 0.0)),
            "top_5_accuracy": metrics.get("top5_accuracy", metrics.get("top4_accuracy", 0.0)),
        }

    return {
        "best_model": best_model,
        "best_score": max(best_score, 0.0),
        "metric_used": "top1_accuracy",
        "all_results": all_results,
        "dataset": {
            "class_count": meta.get("n_classes"),
            "validation_samples": meta.get("val_samples"),
            "target": meta.get("target"),
        },
        "meta": meta,
    }


@visualization_bp.route("/model-info", methods=["GET"])
def get_model_info():
    try:
        report_path = _resolve_report_path()
        if report_path:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return jsonify({"success": True, "data": _build_model_info_payload(report)})
        else:
            return jsonify({"success": False, "error": "模型评估报告尚未生成"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@visualization_bp.route("/stats", methods=["GET"])
def get_stats():
    try:
        stats_path = os.path.join(_resolve_visualization_dir(), "data_exploration_report.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            return jsonify({"success": True, "data": stats})
        else:
            return jsonify({"success": False, "error": "数据探索报告尚未生成"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@visualization_bp.route("/visualization/<path:filename>", methods=["GET"])
def get_visualization(filename):
    try:
        file_path = os.path.join(_resolve_visualization_dir(), filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype="image/png")
        else:
            return jsonify({"success": False, "error": f"文件不存在: {filename}"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
