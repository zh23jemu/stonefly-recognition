from flask import Blueprint, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

predict_bp = Blueprint("predict", __name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "best_stonefly_model.pkl"
)
PREPROCESSOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "preprocessor.pkl"
)
LASSO_SELECTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "lasso_selector.pkl"
)
STAGE1_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "saved_models",
    "stage1_known_unknown_model.pkl",
)
STAGE2_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "saved_models",
    "stage2_known_species_model.pkl",
)
TWO_STAGE_META_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models", "two_stage_metadata.json"
)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not os.path.exists(MODEL_PATH):
            return jsonify(
                {"success": False, "error": "模型尚未训练，请先运行 python train.py"}
            ), 503

        if not os.path.exists(PREPROCESSOR_PATH):
            return jsonify(
                {"success": False, "error": "预处理器未找到，请先运行 python train.py"}
            ), 503

        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        stage1_model = (
            joblib.load(STAGE1_MODEL_PATH)
            if os.path.exists(STAGE1_MODEL_PATH)
            else None
        )
        stage2_model = (
            joblib.load(STAGE2_MODEL_PATH)
            if os.path.exists(STAGE2_MODEL_PATH)
            else None
        )
        two_stage_meta = None
        if os.path.exists(TWO_STAGE_META_PATH):
            with open(TWO_STAGE_META_PATH, "r", encoding="utf-8") as f:
                two_stage_meta = json.load(f)

        # Load LASSO selector if exists
        selector = None
        if os.path.exists(LASSO_SELECTOR_PATH):
            selector = joblib.load(LASSO_SELECTOR_PATH)

        input_df = pd.DataFrame([data])

        required_columns = [
            "lat",
            "lon",
            "country",
            "family",
            "body_length_mm",
            "color",
            "head_feature",
        ]
        for col in required_columns:
            if col not in input_df.columns:
                return jsonify({"success": False, "error": f"缺少必要字段: {col}"}), 400

        # Reorder columns to match training order
        input_df = input_df[required_columns]

        try:
            X_processed = preprocessor.transform(input_df)
            # Apply LASSO feature selection if selector exists
            if selector is not None:
                X_processed = selector.transform(X_processed)
        except ValueError as e:
            if (
                "y contains previously unseen labels" in str(e)
                or "unknown category" in str(e).lower()
            ):
                return jsonify(
                    {
                        "success": False,
                        "error": "输入包含训练时未见过的类别值，请检查country/family/color/head_feature的值是否有效",
                    }
                ), 400
            raise e

        unknown_label = "Unknown Stonefly"
        prediction_label = None
        top_3 = []
        confidence = 0.0
        routing_mode = "single_stage"

        if (
            stage1_model is not None
            and stage2_model is not None
            and two_stage_meta is not None
        ):
            routing_mode = "two_stage"
            known_threshold = float(two_stage_meta.get("known_threshold", 0.4))
            unknown_class_index = int(two_stage_meta.get("unknown_class_index"))

            stage1_proba = stage1_model.predict_proba(X_processed)[0]
            stage1_classes = stage1_model.classes_
            known_class_position = int(np.where(stage1_classes == 1)[0][0])
            known_probability = float(stage1_proba[known_class_position])

            stage2_proba = stage2_model.predict_proba(X_processed)[0]
            stage2_labels = stage2_model.classes_
            stage2_order = np.argsort(stage2_proba)[::-1][:3]

            top_3 = [
                {
                    "species": preprocessor.target_encoder.inverse_transform(
                        [int(stage2_labels[idx])]
                    )[0],
                    "probability": float(stage2_proba[idx]),
                }
                for idx in stage2_order
            ]

            if known_probability >= known_threshold:
                best_idx = int(stage2_order[0])
                pred_class_index = int(stage2_labels[best_idx])
                prediction_label = preprocessor.target_encoder.inverse_transform(
                    [pred_class_index]
                )[0]
                confidence = float(stage2_proba[best_idx])
            else:
                prediction_label = unknown_label
                confidence = 1.0 - known_probability
                top_3.insert(
                    0,
                    {
                        "species": unknown_label,
                        "probability": confidence,
                    },
                )
                top_3 = top_3[:3]
        else:
            prediction = model.predict(X_processed)
            prediction_proba = model.predict_proba(X_processed)
            prediction_label = preprocessor.target_encoder.inverse_transform(
                prediction
            )[0]
            class_indices = np.argsort(prediction_proba[0])[::-1][:3]
            for idx in class_indices:
                class_name = preprocessor.target_encoder.inverse_transform([idx])[0]
                prob = prediction_proba[0][idx]
                top_3.append({"species": class_name, "probability": float(prob)})
            confidence = float(prediction_proba[0][prediction[0]])
        if confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        should_review = prediction_label == unknown_label or confidence < 0.4
        fallback_prediction = None
        if prediction_label == unknown_label and confidence < 0.4:
            for item in top_3:
                if item["species"] != unknown_label:
                    fallback_prediction = item["species"]
                    break

        return jsonify(
            {
                "success": True,
                "prediction": prediction_label,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "should_review": should_review,
                "fallback_prediction": fallback_prediction,
                "routing_mode": routing_mode,
                "top_3_predictions": top_3,
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
