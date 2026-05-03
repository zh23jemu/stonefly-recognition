from .data_exploration import explore_data
from .preprocessing import preprocess_data, DataPreprocessor
from .feature_engineering import perform_feature_engineering, FeatureEngineer
from .model_training import train_models, ModelTrainer
from .model_evaluation import evaluate_and_select_best_model, ModelEvaluator
import numpy as np
import os
import json


def run_complete_ml_pipeline(
    data_path="../data/final_stonefly_dataset.csv",
    output_dir="saved_models",
    viz_dir="visualizations",
    cv=5,
    unknown_cap_ratio=0.2,
    random_state=42,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("石蝇分类系统 - 完整机器学习流程")
    print("=" * 60)

    print("\n【步骤 1/5】数据探索...")
    report, df = explore_data(data_path, output_dir=viz_dir)
    print(f"  [OK] 样本数: {report['basic_statistics']['total_samples']}")
    print(f"  [OK] 特征数: {report['basic_statistics']['total_features']}")
    print(f"  [OK] 类别数: {report['target_analysis']['num_classes']}")

    print("\n【步骤 2/5】数据预处理...")
    import pandas as pd

    df = pd.read_csv(data_path, encoding="utf-8")

    unknown_label = "Unknown Stonefly"
    y_series = pd.Series(df["species"])
    class_counts = y_series.value_counts()
    valid_classes = class_counts[class_counts >= 50].index
    mask = y_series.isin(valid_classes)
    df_filtered = df[mask].copy()

    if unknown_label in df_filtered["species"].values and 0 < unknown_cap_ratio < 1:
        unknown_mask = df_filtered["species"] == unknown_label
        unknown_count = int(unknown_mask.sum())
        known_count = int((~unknown_mask).sum())
        max_unknown = int(known_count * unknown_cap_ratio / (1 - unknown_cap_ratio))
        if unknown_count > max_unknown and max_unknown > 0:
            df_unknown = df_filtered[unknown_mask].sample(
                n=max_unknown, random_state=random_state
            )
            df_known = df_filtered[~unknown_mask]
            df_filtered = pd.concat([df_known, df_unknown], axis=0).sample(
                frac=1, random_state=random_state
            )
            df_filtered = df_filtered.reset_index(drop=True)
            actual_unknown_ratio = float(
                (df_filtered["species"] == unknown_label).mean()
            )
            print(
                f"  [OK] Unknown下采样: {unknown_count} -> {max_unknown}, 当前占比 {actual_unknown_ratio:.2%}"
            )

    print(f"  [OK] 过滤后样本数: {len(df_filtered)} (保留{len(valid_classes)}个类别)")

    preprocessor = DataPreprocessor()
    X, y, df_clean = preprocessor.preprocess_pipeline(df_filtered)
    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))
    print(f"  [OK] 处理后样本数: {len(X)}")
    print(f"  [OK] 处理后特征数: {X.shape[1]}")

    print("\n【步骤 3/5】特征工程...")
    engineer = FeatureEngineer(viz_dir)
    results, X_pca, pca_model, selector = engineer.perform_feature_engineering(
        df_clean, X, y
    )
    print(f"  [OK] PCA降维后: {results['pca']['n_components']} 组件")
    print(f"  [OK] LASSO选择特征: {results['lasso']['n_selected']} 个")

    # Apply LASSO feature selection for training
    X_selected = selector.transform(X)
    if hasattr(X_selected, "columns"):
        selected_feature_names = X_selected.columns.tolist()
    else:
        # Get selected feature names from original X
        selected_mask = selector.get_support()
        selected_feature_names = X.columns[selected_mask].tolist()

    print(
        f"  [OK] 使用LASSO选择的 {len(selected_feature_names)} 个特征进行训练: {selected_feature_names}"
    )

    import joblib

    joblib.dump(pca_model, os.path.join(output_dir, "pca_model.pkl"))
    joblib.dump(selector, os.path.join(output_dir, "lasso_selector.pkl"))
    print(f"  [OK] 特征工程模型已保存")

    print("\n【步骤 4/5】模型训练...")
    trainer = ModelTrainer(output_dir)
    X_train, X_test, y_train, y_test = trainer.prepare_data(X_selected, y)
    unknown_class_index = int(
        np.where(preprocessor.target_encoder.classes_ == unknown_label)[0][0]
    )
    models, results = trainer.train_all_models(X_train, y_train, cv=cv)
    trainer.train_two_stage_models(
        X_train, y_train, unknown_class_index=unknown_class_index, known_threshold=0.4
    )
    trainer.save_models()

    two_stage_meta_path = os.path.join(output_dir, "two_stage_metadata.json")
    with open(two_stage_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unknown_class_index": unknown_class_index,
                "unknown_label": unknown_label,
                "known_threshold": 0.4,
                "unknown_cap_ratio": unknown_cap_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.savez(os.path.join(output_dir, "test_data.npz"), X_test=X_test, y_test=y_test)
    print(f"  [OK] 训练集大小: {len(X_train)}")
    print(f"  [OK] 测试集大小: {len(X_test)}")

    print("\n【步骤 5/5】模型评估与选择...")
    evaluator = ModelEvaluator(output_dir, viz_dir)
    evaluator.evaluate_all_models(X_test, y_test)
    comparison_df = evaluator.compare_models()
    best_name, best_model = evaluator.select_best_model()

    print("\n" + "=" * 60)
    print("机器学习流程完成!")
    print(f"最优模型: {best_name.upper()}")
    print(f"模型已保存至: {output_dir}/")
    print(f"可视化结果已保存至: {viz_dir}/")
    print("=" * 60)

    return evaluator, best_name, best_model


if __name__ == "__main__":
    evaluator, best_name, best_model = run_complete_ml_pipeline()
