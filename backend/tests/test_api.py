"""
后端API测试
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """测试健康检查端点"""
    rv = client.get("/health")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data["status"] == "healthy"
    assert "service" in data


def test_features_endpoint(client):
    """测试特征信息端点"""
    rv = client.get("/api/features")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "features" in data
    assert "target" in data
    assert data["target"] == "family"
    assert len(data["features"]) >= 10


def test_predict_with_model(client):
    """测试训练后的科级预测"""
    rv = client.post(
        "/api/predict",
        data=json.dumps(
            {
                "lat": 30.5,
                "lon": -95.0,
                "country": "US",
                "family": "Perlidae",
                "body_length_mm": 15.0,
                "color": "brown",
                "head_feature": "small antenna",
            }
        ),
        content_type="application/json",
    )
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data["success"] == True
    assert "predicted_family" in data
    assert "family_confidence" in data
    assert "top_3_family_predictions" in data


def test_predict_missing_fields(client):
    """测试缺少字段的预测"""
    rv = client.post(
        "/api/predict", data=json.dumps({"lat": 30.5}), content_type="application/json"
    )
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert data["success"] == False


def test_model_info_endpoint(client):
    """测试模型信息端点"""
    rv = client.get("/api/model-info")
    if rv.status_code == 200:
        data = json.loads(rv.data)
        assert "data" in data
    else:
        assert rv.status_code == 404


def test_stats_endpoint(client):
    """测试统计信息端点"""
    rv = client.get("/api/stats")
    if rv.status_code == 200:
        data = json.loads(rv.data)
        assert "data" in data
    else:
        assert rv.status_code == 404
