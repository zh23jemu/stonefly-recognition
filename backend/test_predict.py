import joblib
import pandas as pd
import numpy as np

# 加载模型
preprocessor = joblib.load("saved_models/preprocessor.pkl")
model = joblib.load("saved_models/best_stonefly_model.pkl")
selector = joblib.load("saved_models/lasso_selector.pkl")

# 测试数据 - 新西兰 Gripopterygidae
data = {
    "lat": -41.2,
    "lon": 174.7,
    "country": "NZ",
    "family": "Gripopterygidae",
    "body_length_mm": 12.0,
    "color": "dark",
    "head_feature": "small antenna",
}

input_df = pd.DataFrame([data])
X_processed = preprocessor.transform(input_df)
X_selected = selector.transform(X_processed)
pred = model.predict(X_selected)
proba = model.predict_proba(X_selected)
species = preprocessor.target_encoder.inverse_transform([pred[0]])[0]

print(f"预测结果: {species}")
print(f"置信度: {proba[0][pred[0]]:.2%}")
print()
print("Top 3 预测:")
top3_idx = np.argsort(proba[0])[::-1][:3]
for idx in top3_idx:
    sp = preprocessor.target_encoder.inverse_transform([idx])[0]
    prob = proba[0][idx]
    print(f"  {sp}: {prob:.2%}")
