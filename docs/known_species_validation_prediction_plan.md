# 删除 Unknown 并改造验证集预测展示

## Summary
- 原始 CSV 不改动；训练流程中先过滤 `species == "Unknown Stonefly"`，再保留样本数不少于 50 的已知物种。
- 数据按 70/15/15 分成训练集、测试集、验证集，随机种子固定为 `42`。
- 预测页改为“从验证集选择一条样本”，右侧展示 KNN、Random Forest、SVM、XGBoost 四个模型分别预测出的物种，并显示真实物种与是否预测正确。
- 训练集最多使用 2 万条做模型训练；测试集和验证集保留完整划分结果，用于评估和页面演示。

## Key Changes
- 后端训练流程：
  - 在训练入口中过滤掉 `Unknown Stonefly`，不再训练两阶段 Unknown/Known 模型。
  - 先划分原始数据，再只用训练集拟合预处理器、PCA/LASSO 选择器，避免测试集/验证集信息泄漏。
  - 保存 `test_data.npz` 用于评估，新增保存验证集样本文件，用于前端选择样本。
  - 模型仍保存四个核心模型：`knn_model.pkl`、`random_forest_model.pkl`、`svm_model.pkl`、`xgboost_model.pkl`。
  - `best_stonefly_model.pkl` 仍按 `macro_f1` 从四个模型中选择，供模型性能页展示最优模型。

- 后端接口：
  - 新增 `GET /api/validation-samples`，分页返回验证集样本、真实物种和输入特征。
  - 调整 `POST /api/predict`，返回四个模型的预测结果数组：
    - `model`
    - `prediction`
    - `confidence`
    - `top_3_predictions`
    - `actual_species`
    - `correct`
  - `predict.py` 不再读取旧的 `stage1_known_unknown_model.pkl`、`stage2_known_species_model.pkl`，避免旧 Unknown 模型影响新结果。

- 前端页面：
  - `PredictView.vue` 左侧改成验证集样本选择区，支持按物种/关键词筛选和分页选择。
  - 右侧改成四模型结果对比表，显示真实物种、预测物种、置信度、Top-3 和正确/错误标记。
  - 移除原来的手动输入表单作为主流程。
  - `DashboardView.vue` 去掉或弱化“未知预测占比”相关展示，改为展示“已过滤 Unknown”、训练/测试/验证样本量、已知物种数等信息。

## Public API / Types
- `PredictionResponse` 改为以 `model_predictions` 为核心字段，不再只表示单个最佳模型结果。
- 新增前端 API 方法 `getValidationSamples(params)`。
- `getModelInfo()` 返回的报告中不再依赖 `unknown_prediction_rate`、`known_only_accuracy`；页面需要兼容这些字段为空或不存在。
- 训练命令新增 `--max-train-samples`，默认 `20000`；Unknown 默认过滤，不再使用 `--unknown-cap-ratio` 作为主逻辑。

## Test Plan
- 后端数据处理测试：
  - 确认训练后的类别中不存在 `Unknown Stonefly`。
  - 确认训练/测试/验证比例约为 70/15/15，且三者样本互不重叠。
  - 确认验证集样本文件包含原始输入特征和真实 `species`。

- 后端接口测试：
  - `GET /api/validation-samples` 能返回分页样本。
  - 对一条验证集样本调用 `POST /api/predict`，必须返回四个模型结果。
  - 每个模型结果都包含预测物种、置信度、Top-3 和 `correct` 字段。
  - 旧的两阶段模型文件即使存在，也不会参与预测。

- 前端验证：
  - 运行前端构建，确认 TypeScript 类型通过。
  - 打开预测页，选择验证集样本后右侧正常显示四个模型结果。
  - 检查桌面和移动宽度下表格、物种名、按钮、卡片不重叠。

- 训练验证命令使用项目本地虚拟环境，例如：
  - `C:\Coding\stonefly-recognition\.venv\Scripts\python.exe backend\train.py --cv 5 --max-train-samples 20000`

## Assumptions
- `Unknown Stonefly` 是唯一需要过滤的 Unknown 标签。
- 不直接删除原始 CSV，也不直接删除旧模型文件；旧两阶段模型保留在磁盘上但不再被接口使用。
- 默认保留“每类至少 50 条”的现有规则。
- 默认划分比例固定为 70/15/15，随机种子固定为 `42`。
