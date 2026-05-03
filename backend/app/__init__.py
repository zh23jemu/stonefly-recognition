"""
Flask应用工厂
"""

from flask import Flask, jsonify
from flask_cors import CORS

from .routes.predict import predict_bp
from .routes.visualization import visualization_bp


def create_app(config=None):
    """创建Flask应用实例"""
    app = Flask(__name__)

    # 启用CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    # 注册蓝图
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(visualization_bp, url_prefix="/api")

    @app.route("/health", methods=["GET"])
    def health_check():
        """健康检查端点"""
        return jsonify(
            {"status": "healthy", "service": "石蝇分类系统", "version": "1.0.0"}
        )

    @app.route("/")
    def index():
        """首页"""
        return jsonify(
            {
                "message": "欢迎使用石蝇分类系统API",
                "docs": "/api/docs",
                "health": "/health",
            }
        )

    return app
