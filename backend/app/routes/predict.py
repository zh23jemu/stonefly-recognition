from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import json

predict_bp = Blueprint("predict", __name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
LASSO_SELECTOR_PATH = os.path.join(MODELS_DIR, "lasso_selector.pkl")
VALIDATION_SAMPLES_PATH = os.path.join(MODELS_DIR, "validation_samples.json")

MODEL_FILES = {
    "knn": "knn_model.pkl",
    "random_forest": "random_forest_model.pkl",
    "svm": "svm_model.pkl",
    "xgboost": "xgboost_model.pkl",
}

REQUIRED_COLUMNS = [
    "lat",
    "lon",
    "country",
    "family",
    "body_length_mm",
    "color",
    "head_feature",
]


def _load_preprocessing_assets():
    """加载训练后保存的预处理器和LASSO选择器。

    预测接口和验证集样本接口都依赖这些产物。如果用户还没有重新训练，
    这里会返回明确错误，避免继续使用旧的Unknown两阶段模型。
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "预处理器未找到，请先使用 .venv\\Scripts\\python.exe backend\\train.py 重新训练"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    selector = joblib.load(LASSO_SELECTOR_PATH) if os.path.exists(LASSO_SELECTOR_PATH) else None
    return preprocessor, selector


def _load_models():
    """加载四个核心分类模型，返回模型字典和缺失模型列表。"""
    models = {}
    missing = []
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
        else:
            missing.append(filename)
    return models, missing


def _prepare_features(data, preprocessor, selector):
    """把原始特征转换成模型输入特征矩阵。"""
    input_df = pd.DataFrame([data])
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要字段: {', '.join(missing_columns)}")

    input_df = input_df[REQUIRED_COLUMNS]
    # 训练流程现在使用完整7个输入特征；LASSO选择器只作为分析报告保留。
    # 因此预测接口不再用selector过滤特征，避免线上输入维度和新模型不一致。
    return preprocessor.transform(input_df)


def _label_from_encoded(preprocessor, encoded_label):
    """将模型输出的数字标签转换回石蝇物种名称。"""
    return preprocessor.target_encoder.inverse_transform([int(encoded_label)])[0]


def _top_predictions(model, X_processed, preprocessor, limit=3):
    """生成单个模型的Top-N预测结果。"""
    if not hasattr(model, "predict_proba"):
        prediction = model.predict(X_processed)[0]
        return [
            {
                "species": _label_from_encoded(preprocessor, prediction),
                "probability": 1.0,
            }
        ]

    probabilities = model.predict_proba(X_processed)[0]
    class_labels = getattr(model, "classes_", np.arange(len(probabilities)))
    order = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "species": _label_from_encoded(preprocessor, class_labels[index]),
            "probability": float(probabilities[index]),
        }
        for index in order
    ]


@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        actual_species = data.get("species") or data.get("actual_species")

        preprocessor, selector = _load_preprocessing_assets()
        models, missing = _load_models()
        if missing:
            return jsonify(
                {
                    "success": False,
                    "error": f"模型文件缺失，请先重新训练: {', '.join(missing)}",
                }
            ), 503

        X_processed = _prepare_features(data, preprocessor, selector)
        model_predictions = []

        for model_name, model in models.items():
            top_3 = _top_predictions(model, X_processed, preprocessor)
            prediction = top_3[0]["species"]
            confidence = top_3[0]["probability"]
            model_predictions.append(
                {
                    "model": model_name,
                    "prediction": prediction,
                    "confidence": confidence,
                    "top_3_predictions": top_3,
                    "actual_species": actual_species,
                    "correct": prediction == actual_species if actual_species else None,
                }
            )

        return jsonify(
            {
                "success": True,
                "actual_species": actual_species,
                "model_predictions": model_predictions,
            }
        )

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@predict_bp.route("/validation-samples", methods=["GET"])
def get_validation_samples():
    try:
        if not os.path.exists(VALIDATION_SAMPLES_PATH):
            return jsonify(
                {
                    "success": False,
                    "error": "验证集样本尚未生成，请先重新训练模型",
                }
            ), 404

        page = max(int(request.args.get("page", 1)), 1)
        page_size = min(max(int(request.args.get("page_size", 10)), 1), 100)
        keyword = request.args.get("keyword", "").strip().lower()
        species = request.args.get("species", "").strip()

        with open(VALIDATION_SAMPLES_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)

        species_options = sorted({item.get("species") for item in samples if item.get("species")})
        if species:
            samples = [item for item in samples if item.get("species") == species]
        if keyword:
            samples = [
                item
                for item in samples
                if keyword in str(item.get("species", "")).lower()
                or keyword in str(item.get("country", "")).lower()
                or keyword in str(item.get("family", "")).lower()
            ]

        total = len(samples)
        start = (page - 1) * page_size
        end = start + page_size
        page_samples = samples[start:end]

        response_samples = []
        for index, item in enumerate(page_samples, start=start):
            response_samples.append(
                {
                    "id": index,
                    "species": item.get("species"),
                    "features": {col: item.get(col) for col in REQUIRED_COLUMNS},
                }
            )

        return jsonify(
            {
                "success": True,
                "page": page,
                "page_size": page_size,
                "total": total,
                "species_options": species_options,
                "samples": response_samples,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@predict_bp.route("/features", methods=["GET"])
def get_features():
    features = {
        "features": [
            {
                "name": "lat",
                "type": "float",
                "description": "纬度",
                "range": "-90 到 90",
            },
            {
                "name": "lon",
                "type": "float",
                "description": "经度",
                "range": "-180 到 180",
            },
            {
                "name": "country",
                "type": "categorical",
                "description": "国家代码",
                "options": ["US", "NZ", "CA", "DE", "NL", "ES", "FR"],
            },
            {
                "name": "family",
                "type": "categorical",
                "description": "石蝇科",
                "options": [
                    "Perlidae",
                    "Austroperlidae",
                    "Pteronarcyidae",
                    "Capniidae",
                    "Leuctridae",
                    "Taeniopterygidae",
                    "Nemouridae",
                    "Eustheniidae",
                    "Gripopterygidae",
                    "Perlodidae",
                ],
            },
            {
                "name": "body_length_mm",
                "type": "float",
                "description": "体长(毫米)",
                "range": "0-50",
            },
            {
                "name": "color",
                "type": "categorical",
                "description": "颜色",
                "options": ["brown", "dark", "yellow", "black"],
            },
            {
                "name": "head_feature",
                "type": "categorical",
                "description": "头部特征",
                "options": ["small antenna", "rounded", "large eye"],
            },
        ],
        "target": "species",
    }
    return jsonify(features)
