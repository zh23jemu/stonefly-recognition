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
HIERARCHICAL_MODEL_PATH = os.path.join(MODELS_DIR, "hierarchical_model_bundle.pkl")

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

HIERARCHICAL_INPUT_COLUMNS = [
    "lat",
    "lon",
    "country",
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


def _load_hierarchical_model():
    """加载 family -> species 层级模型。

    页面主预测现在依赖层级模型。如果用户尚未重新训练，直接返回清晰错误，
    避免悄悄退回旧四模型结果造成展示口径混乱。
    """
    if not os.path.exists(HIERARCHICAL_MODEL_PATH):
        raise FileNotFoundError("层级模型未找到，请先重新训练生成 hierarchical_model_bundle.pkl")
    return joblib.load(HIERARCHICAL_MODEL_PATH)


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


def _prepare_hierarchical_features(data, model_bundle):
    """把页面输入转换为层级模型特征。

    物种子模型仍然只使用 family 之外的 6 个特征。family 由用户先选择，
    用来决定进入哪个 species 子模型，不作为特征列参与模型计算。
    """
    input_df = pd.DataFrame([data])
    missing_columns = [col for col in HIERARCHICAL_INPUT_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要字段: {', '.join(missing_columns)}")
    return model_bundle["preprocessor"].transform(input_df)


def _require_selected_family(data):
    """读取用户选择的 family，并给出明确的缺失提示。"""
    selected_family = data.get("family") or data.get("selected_family")
    if not selected_family:
        raise ValueError("请先选择 family，再在该 family 下预测物种")
    return str(selected_family)


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


def _predict_hierarchical(data, model_bundle, limit=4):
    X_processed = _prepare_hierarchical_features(data, model_bundle)
    selected_family = _require_selected_family(data)

    family_bundle = model_bundle["species_models"].get(selected_family)
    fallback_species_by_family = model_bundle.get("fallback_species_by_family", {})
    global_fallback_species = model_bundle.get("global_fallback_species")
    fallback_species = fallback_species_by_family.get(selected_family, global_fallback_species)
    used_fallback = False
    fallback_reason = None

    if family_bundle:
        species_model = family_bundle["model"]
        species_encoder = family_bundle["species_encoder"]
        species_proba = species_model.predict_proba(X_processed)[0]
        species_class_labels = getattr(
            species_model, "classes_", np.arange(len(species_proba))
        )
        species_order = np.argsort(species_proba)[::-1][:limit]
        top_species = [
            {
                "species": species_encoder.inverse_transform(
                    [int(species_class_labels[index])]
                )[0],
                "probability": float(species_proba[index]),
            }
            for index in species_order
        ]
    else:
        used_fallback = True
        fallback_reason = "family_has_no_species_model"
        top_species = [{"species": fallback_species, "probability": 1.0}]

    actual_family = data.get("family") or data.get("actual_family")
    actual_species = data.get("species") or data.get("actual_species")
    species_prediction = top_species[0]["species"]

    return {
        "selected_family": selected_family,
        "predicted_family": selected_family,
        "family_confidence": 1.0,
        "species_prediction": species_prediction,
        "species_confidence": top_species[0]["probability"],
        "top_4_species_predictions": top_species,
        "actual_family": actual_family,
        "actual_species": actual_species,
        "family_correct": selected_family == actual_family if actual_family else None,
        "species_correct": species_prediction == actual_species if actual_species else None,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
    }


@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        model_bundle = _load_hierarchical_model()
        hierarchical_prediction = _predict_hierarchical(data, model_bundle)

        return jsonify(
            {
                "success": True,
                "prediction_mode": "known_family_species",
                **hierarchical_prediction,
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
