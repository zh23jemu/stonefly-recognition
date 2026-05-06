# 石蝇分类系统 — 项目现状文档

> 更新时间：2026-05-06

---

## 一、项目概述

基于机器学习的石蝇（Plecoptera）物种智能分类系统，毕业设计项目。用户输入观测特征（地理位置、形态、生态信息），系统返回候选物种及概率排名。

**技术栈**：Flask 后端 + Vue 3 前端，Python ML 管道（scikit-learn / XGBoost）

---

## 二、系统架构

```
frontend/              Vue 3 + TypeScript + Element Plus
  └── PredictView      用户输入表单 + 预测结果展示
  └── DashboardView    模型性能可视化大屏

backend/
  ├── app/routes/predict.py     预测 API（POST /api/predict）
  ├── ml/                       ML 核心模块
  │   ├── preprocessing.py      DataPreprocessor（编码/标准化）
  │   ├── model_training.py     ModelTrainer（4 基础模型）
  │   ├── model_evaluation.py   ModelEvaluator（Top-K 评估）
  │   └── hierarchical_training.py  层级模型（Family→Species）
  ├── train.py                  原始训练入口
  ├── train_data2_enhanced.py   data2 增强训练脚本（8 特征）
  └── train_augmented.py        增强数据集训练脚本（11 特征）★ 新增

data/
  ├── final_stonefly_dataset.csv          原始数据（8.8 MB，99,901 条）
  ├── stonefly_combined_data_data2.csv    合并数据（10.4 MB，67,493 条）
  └── stonefly_combined_data_augmented.csv  增强数据（44 MB，278,259 条）★ 新增

scripts/
  ├── slurm_train_augmented.sbatch        正式训练 SLURM 脚本 ★ 新增
  └── slurm_train_augmented_smoke.sbatch  冒烟测试 SLURM 脚本 ★ 新增
```

**预测流程（当前生产）**：
```
用户选择科（Family）→ 后端用 family 过滤 → 层级 XGBoost 在科内预测物种 → 返回 Top-4 候选
```

---

## 三、数据集现状

### 3.1 增强数据集（主力数据集）

文件：`data/stonefly_combined_data_augmented.csv`

| 来源 | 条数 | 说明 |
|------|------|------|
| `data` | 63,997 | 原始模拟数据，有 body_length_mm / color / head_feature |
| `data2` | 3,496 | GBIF 真实观测（Arkansas Plecoptera V1），有 gbifID |
| `synthetic` | 210,766 | 合成补充数据（v2 脚本生成）|
| **合计** | **278,259** | 696 个物种，全部 ≥ 300 条 |

**特征字段（11 个）**：

| 字段 | 类型 | 来源覆盖情况 |
|------|------|------------|
| lat / lon | 数值 | 全部 |
| country | 分类 | 全部 |
| family | 分类 | 全部（16 科） |
| body_length_mm | 数值 | data + synthetic（data2 缺失，中位数填充） |
| color | 分类 | data + synthetic（data2 为 unknown） |
| head_feature | 分类 | data + synthetic（data2 为 unknown） |
| month | 分类 | synthetic（按科级生态历填充）；data/data2 为 unknown |
| habitat | 分类 | synthetic（stream/river/spring/lake）；其余 unknown |
| sex | 分类 | synthetic（male/female 按比例）；其余 unknown |
| life_stage | 分类 | synthetic（adult/immature/egg）；其余 unknown |

**物种分布**：

| 指标 | 数值 |
|------|------|
| 总物种数 | 696 |
| 最少记录物种 | 300 条（`Leuctra variabilis` 等） |
| 最多记录物种 | 4,853 条（`Nemoura cinerea`） |
| 混淆物种（科内 geo_cell 重叠 ≥ 3 格）| 283 个，目标提升至 420 条 |
| 普通物种目标 | 300 条 |

**各科物种数**：

| 科 | 物种数 |
|----|--------|
| Perlidae | 123 |
| Perlodidae | 106 |
| Nemouridae | 101 |
| Leuctridae | 78 |
| Capniidae | 70 |
| Chloroperlidae | 57 |
| Gripopterygidae | 51 |
| Taeniopterygidae | 43 |
| 其余 8 科 | 27 |

### 3.2 合成数据生成策略（v2）

脚本：`scripts/generate_synthetic_data_v2.py`

- **普通物种（≥5 条存量）**：Bootstrap 抽样 + 高斯噪声（lat/lon ±0.4°，体长 ±8%）
- **极少物种（<5 条）**：使用科级均值/标准差生成
- **混淆物种**：50% 的点落在科内重叠 geo_cell 中，加强边界样本
- **生态字段**：按 16 科各自的生态历分配月份、栖息地、性别、生命阶段

---

## 四、模型现状

### 4.1 已部署模型（`backend/saved_models/`）

基于**原始数据集**（`final_stonefly_dataset.csv`）训练，特征 7 个（无 month/habitat/sex/life_stage）。

**平铺 4 基础模型评估结果**（当前线上）：

| 模型 | Top-1 | Top-3 | Top-5 | F1-macro |
|------|-------|-------|-------|----------|
| **XGBoost** ✅ 最优 | **0.4841** | 0.8229 | 0.9358 | 0.4481 |
| Random Forest | 0.4833 | 0.8093 | 0.9192 | 0.4164 |
| KNN | 0.4244 | 0.7195 | 0.8033 | 0.2760 |
| SVM | 0.3094 | 0.7863 | 0.9218 | 0.2710 |

**层级模型评估结果**（Family → Species，当前线上主用）：

| 指标 | 数值 |
|------|------|
| 科级预测 Top-1 | 47.8% |
| 物种级 Top-1（层级链路） | 20.8% |
| 物种级 Top-3 | 37.7% |
| 物种级 Top-5 | 44.4% |
| Oracle 上限（科选对时物种准确率） | 55.9% |

> **注**：上述结果基于原始数据集（类别少、样本少），层级模型准确率偏低的主要原因是科级预测本身准确率不足（47.8%）。当前生产预测流程已改为**用户手动选择科**，绕过了科预测这一瓶颈。

### 4.2 增强数据集训练（集群进行中）

脚本：`backend/train_augmented.py`，目标输出：`backend/saved_models_augmented/`

**配置参数**：

| 模型 | 训练样本 | 关键超参 | 预期 Top-1 |
|------|---------|---------|-----------|
| XGBoost | 全量 ~19.5 万 | trees=300, depth=10, lr=0.05, hist 方法 | TBD |
| RF | 8 万（分层抽样）| trees=600, max_depth=30, n_jobs=8 | TBD |
| KNN | 5 万 | k=7, distance 权重 | TBD |
| SVM | 3 万 | C=10, RBF 核（上限 3 万，受 O(n²) 限制） | TBD |

**已知训练耗时参考**（集群 32 CPU，80G 内存）：
- XGBoost 1500 棵 × 696 类：~54 分钟（内存 ~18 GB）
- XGBoost 300 棵：估计 ~10-15 分钟

**OOM 问题（已修复）**：

| 次序 | 原因 | 修复方案 |
|------|------|---------|
| 第 1 次 | RF `max_depth=None` + 15 万样本，叶节点爆炸 | 添加 `--rf-max-depth 30`，RF 样本降至 8 万 |
| 第 2 次 | XGBoost 训练完后模型留在内存（~15-20 GB），RF 启动时 OOM | 每个模型训练+评估+保存后，`del model; gc.collect()` |

---

## 五、准确率问题分析

**当前最优（XGBoost）Top-1 = 48%，目标 90%，差距较大。**

根本原因：

1. **类别数过多**：696 个物种平铺分类，即使每类 280 条训练样本，特征区分度不足
2. **特征区分度弱**：`color` 仅 4 值，`head_feature` 仅 3 值，许多物种形态完全相同
3. **地理重叠**：283 个物种在 geo_cell 层面重叠，lat/lon 无法区分它们
4. **合成数据边界模糊**：合成数据基于原有记录 bootstrap 生成，相似物种间人为边界本来就模糊

**增强数据集能否显著提升？**

增强数据将训练样本从 ~4.5 万提升至 ~19.5 万，且加入了 month/habitat/sex/life_stage 特征，预期有一定提升，但平铺 696 类下 90% Top-1 可能性极低。

**达到 90% 的可行路径**：

| 方案 | 原理 | 预期效果 |
|------|------|---------|
| 用户选科 + 科内分类（已实现） | 每科 10-100 物种，问题规模大幅缩小 | 科内 Top-1 预计 70-90% |
| 缩减目标物种数（如 Top-100） | 只保留最常见物种，减少混淆 | 平铺 Top-1 可达 70%+ |
| 引入更多区分特征（翅脉、DNA 等） | 特征本身不足是瓶颈 | 理论上可达 90%+ |

---

## 六、待办事项

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 🔴 高 | 等待集群 OOM 修复后的训练结果 | 进行中 |
| 🔴 高 | 评估增强数据集 4 模型的实际精度 | 待训练完成 |
| 🟡 中 | 用增强数据集重训层级模型（用户手选科 → 科内分类） | 待做 |
| 🟡 中 | 前端 PredictView 支持 month/habitat/sex 可选输入字段 | 待做 |
| 🟡 中 | predict.py API 切换到 saved_models_augmented/ 新模型 | 待训练完成后 |
| 🟢 低 | 评估缩减物种数（Top-N）对精度的影响 | 可选 |
| 🟢 低 | 整理实验对比报告（原始 vs 增强数据集） | 可选 |

---

## 七、关键文件速查

| 文件 | 用途 |
|------|------|
| `data/stonefly_combined_data_augmented.csv` | 主力训练数据（278k 条，696 物种）|
| `backend/train_augmented.py` | 增强数据集训练脚本（11 特征，4 模型）|
| `scripts/slurm_train_augmented.sbatch` | 正式训练 SLURM 脚本（32 CPU，80G，12h）|
| `scripts/slurm_train_augmented_smoke.sbatch` | 冒烟测试脚本（8 CPU，1h）|
| `scripts/generate_synthetic_data_v2.py` | 合成数据生成脚本（v2，含生态历）|
| `backend/saved_models/hierarchical_model_bundle.pkl` | 当前线上层级模型 |
| `backend/app/routes/predict.py` | 预测 API 入口 |

---

## 八、本地启动方式

```bash
# 后端（端口 5000）
cd backend
.venv/Scripts/python.exe run.py

# 前端（端口 5173）
cd frontend
npm run dev
```

访问：http://localhost:5173

---

## 九、集群训练命令参考

```bash
# 冒烟测试（验证流程，约 15 分钟）
sbatch scripts/slurm_train_augmented_smoke.sbatch

# 正式训练（全部 4 模型，约 2-4 小时）
sbatch scripts/slurm_train_augmented.sbatch

# 跳过 SVM（最慢），约 1-2 小时
sbatch --export=ALL,SKIP_SVM=1 scripts/slurm_train_augmented.sbatch

# 查看日志
tail -f logs/slurm/stonefly-augmented_<job_id>.out
```
