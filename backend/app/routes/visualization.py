from flask import Blueprint, jsonify, send_file
import os
import json

visualization_bp = Blueprint("visualization", __name__)

VISUALIZATIONS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "visualizations"
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models")


@visualization_bp.route("/model-info", methods=["GET"])
def get_model_info():
    try:
        report_path = os.path.join(MODELS_DIR, "model_evaluation_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return jsonify({"success": True, "data": report})
        else:
            return jsonify({"success": False, "error": "模型评估报告尚未生成"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@visualization_bp.route("/stats", methods=["GET"])
def get_stats():
    try:
        stats_path = os.path.join(VISUALIZATIONS_DIR, "data_exploration_report.json")
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
        file_path = os.path.join(VISUALIZATIONS_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype="image/png")
        else:
            return jsonify({"success": False, "error": f"文件不存在: {filename}"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
