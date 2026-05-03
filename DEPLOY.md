# 石蝇分类系统 - 部署指南

## 系统要求

### 后端要求
- Python 3.8+
- 8GB+ RAM（推荐16GB用于大数据集训练）
- 5GB+ 磁盘空间

### 前端要求
- **Node.js 18.x 或 20.x** (推荐 18.20.0, 不支持 Node 22+)
- npm 9+

**注意**: 项目包含 `.nvmrc` 文件，可使用 nvm 自动切换 Node 版本:
```bash
cd frontend
nvm use  # 自动切换到 Node 18.20.0
```

## 安装步骤

### 1. 环境准备

确保已安装：
- Python 3.8或更高版本
- **Node.js 18.x 或 20.x** (推荐 18.20.0)
- Git（可选）

### 2. 安装后端依赖

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
cd frontend
npm install
```

### 4. 训练模型（首次运行必须）

```bash
cd backend
python train.py --cv 5 --unknown-cap-ratio 0.20
```

可选参数说明：
- `--cv`：交叉验证折数（默认5）
- `--unknown-cap-ratio`：Unknown 类下采样后最大占比（默认0.20）

例如：
```bash
# 更激进地减少 Unknown
python train.py --cv 5 --unknown-cap-ratio 0.15

# 更保守
python train.py --cv 5 --unknown-cap-ratio 0.25
```

### 4.1 Unknown 占比三档对比实验（答辩建议）

```bash
cd backend
python experiments/unknown_cap_sweep.py --caps 0.15 0.20 0.25 --cv 3
```

生成报告位置：
- `backend/visualizations/unknown_cap_sweep_*.json`
- `backend/visualizations/unknown_cap_sweep_*.csv`
- `backend/visualizations/unknown_cap_sweep_*.md`

训练过程包括：
- 数据探索和分析
- 数据预处理
- 特征工程（PCA、LASSO）
- 4种模型训练（随机森林、SVM、XGBoost、KNN）
- 模型评估和选择

训练完成后，最优模型将保存在 `backend/saved_models/` 目录。

### 5. 启动服务

#### 方法1：使用启动脚本（推荐）

```bash
# 启动后端
start.bat backend

# 启动前端
start.bat frontend

# 同时启动所有服务
start.bat all
```

#### 方法2：手动启动

终端1（后端）：
```bash
cd backend
venv\Scripts\activate
python run.py
```

终端2（前端）：
```bash
cd frontend
npm run dev
```

## 访问系统

- 前端界面：http://localhost:5173
- 后端API：http://localhost:5000
- API文档：http://localhost:5000/api/features

## 项目结构说明

```
stonefly-classification/
├── backend/
│   ├── app/                    # Flask应用
│   │   ├── __init__.py        # 应用工厂
│   │   ├── routes/            # API路由
│   │   │   ├── predict.py     # 预测接口
│   │   │   └── visualization.py  # 可视化接口
│   │   ├── services/          # 业务逻辑服务
│   │   ├── models/            # 数据模型
│   │   └── utils/             # 工具函数
│   ├── ml/                     # 机器学习模块
│   │   ├── __init__.py        # 完整流程入口
│   │   ├── data_exploration.py   # 数据探索
│   │   ├── preprocessing.py   # 数据预处理
│   │   ├── feature_engineering.py  # 特征工程
│   │   ├── model_training.py  # 模型训练
│   │   └── model_evaluation.py   # 模型评估
│   ├── saved_models/          # 保存的模型文件
│   ├── visualizations/        # 可视化图表
│   ├── config.yaml            # 配置文件
│   ├── requirements.txt       # Python依赖
│   ├── run.py                 # 后端启动脚本
│   └── train.py               # 模型训练脚本
├── frontend/
│   ├── src/                   # Vue源码
│   │   ├── components/        # 可复用组件
│   │   ├── views/             # 页面视图
│   │   │   ├── HomeView.vue       # 首页
│   │   │   ├── PredictView.vue    # 预测页面
│   │   │   └── DashboardView.vue  # 数据大屏
│   │   ├── services/          # API服务
│   │   ├── router/            # 路由配置
│   │   ├── App.vue            # 根组件
│   │   └── main.ts            # 入口文件
│   ├── package.json           # Node依赖
│   ├── tsconfig.json          # TypeScript配置
│   └── vite.config.ts         # Vite配置
├── data/                      # 数据集
│   └── final_stonefly_dataset.csv
├── start.bat                  # Windows启动脚本
├── README.md                  # 项目说明
└── DEPLOY.md                  # 部署指南
```

## API接口文档

### 1. 预测接口

**URL**: `POST /api/predict`

**请求体**:
```json
{
  "lat": 30.5,
  "lon": -95.0,
  "country": "US",
  "family": "Perlidae",
  "body_length_mm": 15.0,
  "color": "brown",
  "head_feature": "small antenna"
}
```

**响应**:
```json
{
  "success": true,
  "prediction": "Paragnetina fumosa",
  "confidence": 0.95,
  "top_3_predictions": [
    {"species": "Paragnetina fumosa", "probability": 0.95},
    {"species": "Acroneuria abnormis", "probability": 0.03},
    {"species": "Other", "probability": 0.02}
  ]
}
```

### 2. 获取特征信息

**URL**: `GET /api/features`

**响应**:
```json
{
  "features": [
    {"name": "lat", "type": "float", "description": "纬度", "range": "-90 到 90"},
    {"name": "lon", "type": "float", "description": "经度", "range": "-180 到 180"},
    {"name": "country", "type": "categorical", "description": "国家代码"},
    {"name": "family", "type": "categorical", "description": "石蝇科"},
    {"name": "body_length_mm", "type": "float", "description": "体长(毫米)", "range": "0-50"},
    {"name": "color", "type": "categorical", "description": "颜色"},
    {"name": "head_feature", "type": "categorical", "description": "头部特征"}
  ],
  "target": "species"
}
```

### 3. 获取模型信息

**URL**: `GET /api/model-info`

**响应**:
```json
{
  "success": true,
  "data": {
    "best_model": "xgboost",
    "best_score": 0.92,
    "metric_used": "f1_score",
    "all_results": {
      "random_forest": {"accuracy": 0.90, "precision": 0.89, "recall": 0.90, "f1_score": 0.89},
      "svm": {"accuracy": 0.88, "precision": 0.87, "recall": 0.88, "f1_score": 0.87},
      "xgboost": {"accuracy": 0.92, "precision": 0.91, "recall": 0.92, "f1_score": 0.92},
      "knn": {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1_score": 0.84}
    }
  }
}
```

### 4. 获取可视化图表

**URL**: `GET /api/visualization/{filename}`

支持的图表文件：
- `model_comparison.png` - 模型对比图
- `class_distribution.png` - 类别分布图
- `feature_distributions.png` - 特征分布图
- `correlation_matrix.png` - 相关矩阵图
- `confusion_matrix_{model}.png` - 混淆矩阵

## 功能模块说明

### 模块1: 数据探索
- 数据集统计分析
- 缺失值检测
- 类别分布分析
- 异常值检测

### 模块2: 数据预处理
- 缺失值填充（中位数/众数）
- 异常值处理（IQR方法）
- 分类变量编码（LabelEncoder）
- 特征标准化（StandardScaler）

### 模块3: 特征工程
- 探索性数据分析（EDA）
- 特征分布可视化
- 相关系数计算
- PCA主成分分析降维
- LASSO回归特征选择

### 模块4: 模型训练
- 随机森林（Random Forest）
- 支持向量机（SVM）
- XGBoost
- K近邻（KNN）
- 超参数调优（网格搜索 + 交叉验证）

### 模块5: 模型评估
- 准确率、精确率、召回率、F1分数
- 混淆矩阵可视化
- 模型性能对比
- 最优模型选择

### 模块6: Web系统
- 石蝇特征输入界面
- 分类结果展示（Top 3）
- 模型性能可视化大屏
- RESTful API接口

## 故障排除

### 1. 训练时内存不足
```bash
# 减少训练数据量或交叉验证折数
# 修改 backend/ml/__init__.py 中的 cv 参数
cv=2  # 原为 cv=3
```

### 2. 前端无法连接后端
- 检查后端服务是否运行在端口5000
- 检查CORS配置是否正确
- 查看浏览器控制台网络请求

### 3. 模型加载失败
- 确保已运行训练脚本 `python train.py`
- 检查 `backend/saved_models/` 目录是否存在模型文件

### 4. 依赖安装失败
```bash
# 更新pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 性能优化

### 训练加速
- 使用更多CPU核心（已设置n_jobs=-1）
- 减少交叉验证折数
- 缩小超参数搜索范围

### 推理优化
- 使用轻量级模型（如KNN）
- 缓存预测结果
- 批量预测接口

## 安全注意事项

1. 生产环境部署时：
   - 禁用Flask调试模式
   - 使用HTTPS
   - 添加身份验证
   - 限制API访问频率

2. 修改 `backend/config.yaml`：
```yaml
app:
  debug: false
  host: "127.0.0.1"
```

## 技术支持

如有问题，请检查：
1. README.md 文档
2. 浏览器开发者工具（前端问题）
3. 后端日志输出
4. 模型训练日志
