from .data_exploration import explore_data
from .preprocessing import preprocess_data, DataPreprocessor
from .feature_engineering import perform_feature_engineering, FeatureEngineer
from .model_training import train_models, ModelTrainer
from .model_evaluation import evaluate_and_select_best_model, ModelEvaluator
from .hierarchical_training import train_hierarchical_models
import numpy as np
import os
import json


def run_complete_ml_pipeline(
    data_path="../data/final_stonefly_dataset.csv",
    output_dir="saved_models",
    viz_dir="visualizations",
    cv=5,
    max_train_samples=20000,
    random_state=42,
    train_hierarchical=True,
    hierarchical_n_jobs=2,
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
    valid_classes = class_counts[
        (class_counts >= 50) & (class_counts.index != unknown_label)
    ].index
    mask = y_series.isin(valid_classes) & (y_series != unknown_label)
    df_filtered = df[mask].copy()
    removed_unknown_count = int((y_series == unknown_label).sum())

    print(f"  [OK] 已删除Unknown样本数: {removed_unknown_count}")
    print(f"  [OK] 过滤后样本数: {len(df_filtered)} (保留{len(valid_classes)}个已知类别)")

    from sklearn.model_selection import StratifiedShuffleSplit

    # 先在原始已知物种数据上做分层划分，之后只用训练集拟合预处理器和特征选择器。
    y_filtered = df_filtered["species"].to_numpy()
    split_index = np.arange(len(df_filtered))
    holdout_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state
    )
    train_idx, holdout_idx = next(holdout_splitter.split(split_index, y_filtered))

    df_train = df_filtered.iloc[train_idx].reset_index(drop=True)
    df_holdout = df_filtered.iloc[holdout_idx].reset_index(drop=True)
    y_holdout = df_holdout["species"].to_numpy()
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=random_state
    )
    test_idx, validation_idx = next(
        validation_splitter.split(np.arange(len(df_holdout)), y_holdout)
    )
    df_test = df_holdout.iloc[test_idx].reset_index(drop=True)
    df_validation = df_holdout.iloc[validation_idx].reset_index(drop=True)

    preprocessor = DataPreprocessor()
    X_train_full, y_train_full, df_train_clean = preprocessor.preprocess_pipeline(
        df_train
    )
    X_test_full, y_test, df_test_clean = preprocessor.transform_pipeline(df_test)
    X_validation_full, y_validation, df_validation_clean = (
        preprocessor.transform_pipeline(df_validation)
    )
    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))
    print(f"  [OK] 训练集样本数: {len(X_train_full)}")
    print(f"  [OK] 测试集样本数: {len(X_test_full)}")
    print(f"  [OK] 验证集样本数: {len(X_validation_full)}")
    print(f"  [OK] 处理后特征数: {X_train_full.shape[1]}")

    print("\n【步骤 3/5】特征工程...")
    engineer = FeatureEngineer(viz_dir)
    results, X_pca, pca_model, selector = engineer.perform_feature_engineering(
        df_train_clean, X_train_full, y_train_full
    )
    print(f"  [OK] PCA降维后: {results['pca']['n_components']} 组件")
    print(f"  [OK] LASSO选择特征: {results['lasso']['n_selected']} 个")

    # LASSO结果只作为可视化和报告参考，不再压缩模型训练输入。
    # 当前只有7个手动输入特征，实验显示强行筛掉体长/头部特征会降低Top-1准确率；
    # 因此四个模型统一使用完整特征，提高模型能利用的信息量。
    selected_mask = selector.get_support()
    selected_feature_names = X_train_full.columns[selected_mask].tolist()

    print(
        f"  [OK] 使用LASSO选择的 {len(selected_feature_names)} 个特征进行训练: {selected_feature_names}"
    )

    import joblib

    joblib.dump(pca_model, os.path.join(output_dir, "pca_model.pkl"))
    joblib.dump(selector, os.path.join(output_dir, "lasso_selector.pkl"))
    print(f"  [OK] 特征工程模型已保存")

    print("\n【步骤 4/5】模型训练...")
    trainer = ModelTrainer(output_dir)
    X_train, y_train = X_train_full, y_train_full
    if max_train_samples and len(X_train) > max_train_samples:
        from sklearn.model_selection import StratifiedShuffleSplit

        sampler = StratifiedShuffleSplit(
            n_splits=1, train_size=max_train_samples, random_state=random_state
        )
        sampled_idx, _ = next(sampler.split(np.zeros(len(y_train)), y_train))
        X_train = (
            X_train.iloc[sampled_idx]
            if hasattr(X_train, "iloc")
            else X_train[sampled_idx]
        )
        y_train = np.asarray(y_train)[sampled_idx]
        print(f"  [OK] 训练集已分层抽样到: {len(X_train)}")

    models, results = trainer.train_all_models(X_train, y_train, cv=cv)
    trainer.save_models()

    np.savez(
        os.path.join(output_dir, "test_data.npz"),
        X_test=X_test_full,
        y_test=y_test,
    )
    np.savez(
        os.path.join(output_dir, "validation_data.npz"),
        X_validation=X_validation_full,
        y_validation=y_validation,
    )
    validation_samples_path = os.path.join(output_dir, "validation_samples.json")
    validation_records = df_validation_clean.to_dict(orient="records")
    with open(validation_samples_path, "w", encoding="utf-8") as f:
        json.dump(validation_records, f, ensure_ascii=False, indent=2)

    split_meta_path = os.path.join(output_dir, "dataset_split_metadata.json")
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unknown_policy": "removed",
                "unknown_label": unknown_label,
                "removed_unknown_count": removed_unknown_count,
                "min_class_count": 50,
                "split_ratio": {"train": 0.70, "test": 0.15, "validation": 0.15},
                "max_train_samples": max_train_samples,
                "class_count": int(len(valid_classes)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test_full)),
                "validation_samples": int(len(X_validation_full)),
                "model_feature_policy": "full_7_features_without_lasso_filter",
                "random_state": random_state,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  [OK] 训练集大小: {len(X_train)}")
    print(f"  [OK] 测试集大小: {len(X_test_full)}")
    print(f"  [OK] 验证集大小: {len(X_validation_full)}")

    print("\n【步骤 5/5】模型评估与选择...")
    evaluator = ModelEvaluator(output_dir, viz_dir)
    evaluator.evaluate_all_models(X_test_full, y_test)
    comparison_df = evaluator.compare_models()
    best_name, best_model = evaluator.select_best_model()

    if train_hierarchical:
        print("\n【附加步骤】训练 family -> species 层级模型...")
        hierarchical_results = train_hierarchical_models(
            data_path=data_path,
            output_dir=output_dir,
            max_train_samples=max_train_samples,
            random_state=random_state,
            n_jobs=hierarchical_n_jobs,
        )
        print("  [OK] 层级模型结果已保存到 hierarchical_results.json")
        print(
            "  [OK] 层级Species Top-1: "
            f"{hierarchical_results['metrics']['hierarchical_species_accuracy']:.4f}"
        )
        print(
            "  [OK] 层级Species Top-4: "
            f"{hierarchical_results['metrics']['hierarchical_species_top_4_accuracy']:.4f}"
        )

    print("\n" + "=" * 60)
    print("机器学习流程完成!")
    print(f"最优模型: {best_name.upper()}")
    print(f"模型已保存至: {output_dir}/")
    print(f"可视化结果已保存至: {viz_dir}/")
    print("=" * 60)

    return evaluator, best_name, best_model


if __name__ == "__main__":
    evaluator, best_name, best_model = run_complete_ml_pipeline()
