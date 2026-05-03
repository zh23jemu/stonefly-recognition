# 石蝇分类系统

基于机器学习的石蝇智能分类系统，采用随机森林、SVM、XGBoost和KNN等多种算法，结合PCA降维和LASSO特征选择，实现高精度的石蝇种类分类。

## 系统架构

- **后端**: Python + Flask + scikit-learn + XGBoost
- **前端**: Vue.js 3 + TypeScript + Element Plus
- **数据库**: CSV数据集 (99901条记录)

## 功能特性

### 1. 数据预处理
- 缺失值处理（中位数填充）
- 异常值检测与处理（IQR方法）
- 分类变量编码（LabelEncoder）
- 特征标准化/归一化

### 2. 特征工程
- 探索性数据分析（EDA）
- 主成分分析（PCA）降维
- LASSO回归特征选择
- 特征分布可视化
- 相关系数分析

### 3. 机器学习模型
- **随机森林（Random Forest）**
- **支持向量机（SVM）**
- **XGBoost**
- **K近邻（KNN）**

每个模型都经过超参数调优（随机森林使用随机搜索 + 其他使用网格搜索 + 交叉验证）

### 4. 模型评估
- 准确率、精确率、召回率、F1分数
- 混淆矩阵可视化
- ROC曲线和AUC
- 模型性能对比

### 5. Web界面
- 石蝇特征输入表单
- 分类结果展示（Top 3预测）
- 模型性能可视化大屏
- RESTful API接口

## 快速启动

### Windows

```bash
# 启动后端服务
start.bat backend

# 启动前端服务
start.bat frontend

# 同时启动所有服务
start.bat all

# 训练模型
start.bat train
```

### 手动启动

**1. 训练模型（首次使用）**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py --cv 5 --unknown-cap-ratio 0.20
```

**可调参数（建议答辩展示）**
```bash
# Unknown 占比压到 15%（更激进，通常更少 Unknown 预测）
python train.py --cv 5 --unknown-cap-ratio 0.15

# Unknown 占比压到 25%（更保守）
python train.py --cv 5 --unknown-cap-ratio 0.25
```

**自动三档对比实验（生成答辩报告）**
```bash
cd backend
python experiments/unknown_cap_sweep.py --caps 0.15 0.20 0.25 --cv 3
```

实验完成后会自动生成：
- `backend/visualizations/unknown_cap_sweep_*.json`
- `backend/visualizations/unknown_cap_sweep_*.csv`
- `backend/visualizations/unknown_cap_sweep_*.md`

**2. 启动后端**
```bash
cd backend
venv\Scripts\activate
python run.py
```

**3. 启动前端**
```bash
cd frontend
npm install
npm run dev
```

## API接口

### 预测接口
```http
POST /api/predict
Content-Type: application/json

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

### 获取模型信息
```http
GET /api/model-info
```

### 获取统计数据
```http
GET /api/stats
```

## 数据集说明

数据集包含99901条石蝇样本记录，8个特征：
- `species`: 石蝇种类（目标变量）
- `lat`: 纬度
- `lon`: 经度
- `country`: 国家代码
- `family`: 石蝇科
- `body_length_mm`: 体长（毫米）
- `color`: 颜色
- `head_feature`: 头部特征

## 项目结构

```
stonefly-classification/
├── backend/
│   ├── app/              # Flask应用
│   ├── ml/               # 机器学习模块
│   ├── saved_models/     # 保存的模型
│   ├── config.yaml       # 配置文件
│   ├── requirements.txt  # Python依赖
│   └── run.py            # 启动脚本
├── frontend/
│   ├── src/              # Vue源码
│   │   ├── components/   # 组件
│   │   ├── views/        # 页面视图
│   │   └── services/     # API服务
│   ├── package.json      # Node依赖
│   └── vite.config.ts    # Vite配置
├── data/
│   └── final_stonefly_dataset.csv  # 数据集
├── visualizations/       # 可视化结果
├── start.bat            # Windows启动脚本
└── README.md            # 说明文档
```

## 技术栈

### 后端
- Flask 3.0.0
- scikit-learn 1.3.0
- XGBoost 2.0.0
- Pandas 2.1.0
- NumPy 1.26.0

### 前端
- Vue.js 3.4.0
- TypeScript 5.3.0
- Element Plus 2.4.0
- Axios 1.6.0
- Vite 5.0.0
- **Node.js 18.x 或 20.x** (推荐 18.20.0)

## 作者

毕业设计项目 - 基于机器学习的石蝇分类系统

## 许可证

MIT License
