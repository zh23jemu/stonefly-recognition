"""
石蝇分类系统后端启动脚本
"""

from app import create_app
import yaml

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = create_app(config)

if __name__ == "__main__":
    app.run(
        host=config["app"]["host"],
        port=config["app"]["port"],
        debug=config["app"]["debug"],
    )
