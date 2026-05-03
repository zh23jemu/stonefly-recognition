import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
import os


UNKNOWN_CATEGORY = "__UNKNOWN__"


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.scaling_method = "standard"
        self.numeric_features = []
        self.categorical_features = []
        self.target_encoder = None
        self.missing_values = {}
        self.outlier_bounds = {}

    def fit(self, df, target_column="species", scaling_method="standard"):
        self.scaling_method = scaling_method

        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        if target_column in self.numeric_features:
            self.numeric_features.remove(target_column)
        if target_column in self.categorical_features:
            self.categorical_features.remove(target_column)

        for col in self.categorical_features:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            # 为测试集、验证集或页面输入中训练集未见过的分类值保留兜底编码。
            # 这样预处理规则仍然只由训练集拟合，同时不会因为新国家/科/颜色等
            # 分类值直接中断预测流程。
            values = pd.Series(df[col].unique().tolist() + [UNKNOWN_CATEGORY])
            le.fit(values)
            self.label_encoders[col] = le

        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df[target_column])

        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        if self.numeric_features:
            self.scaler.fit(df[self.numeric_features])

        return self

    def fit_cleaning_rules(
        self, df, target_column="species", missing_strategy="median", threshold=1.5
    ):
        """只基于训练集学习缺失值填充值和异常值截断边界。

        训练/测试/验证划分后，测试集和验证集不能反过来影响数据清洗规则；
        因此这里把中位数、众数和 IQR 边界保存下来，后续所有数据集都复用
        训练集上得到的规则，避免评估阶段出现数据泄漏。
        """
        self.missing_values = {}
        self.outlier_bounds = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == target_column:
                continue

            if missing_strategy == "median":
                fill_value = df[col].median()
            elif missing_strategy == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = 0
            self.missing_values[col] = fill_value

            clean_series = df[col].fillna(fill_value)
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = (
                q1 - threshold * iqr,
                q3 + threshold * iqr,
            )

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col == target_column:
                continue
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            self.missing_values[col] = mode_value

    def apply_cleaning_rules(self, df, target_column="species"):
        """使用训练集清洗规则转换任意数据集。

        该方法不会重新计算中位数、众数或异常值边界，保证验证集样本在
        页面演示时经过的预处理流程与模型训练时一致。
        """
        df_clean = df.copy()

        for col, fill_value in self.missing_values.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(fill_value)

        for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(
                    lower=lower_bound, upper=upper_bound
                )

        return df_clean

    def transform(self, df, target_column="species"):
        df_processed = df.copy()

        for col in self.categorical_features:
            df_processed[col] = df_processed[col].astype(str)
            known_values = set(self.label_encoders[col].classes_)
            df_processed[col] = df_processed[col].where(
                df_processed[col].isin(known_values), UNKNOWN_CATEGORY
            )
            df_processed[col] = self.label_encoders[col].transform(df_processed[col])

        if self.numeric_features and self.scaler:
            df_processed[self.numeric_features] = self.scaler.transform(
                df_processed[self.numeric_features]
            )

        if target_column in df_processed.columns:
            if self.target_encoder is None:
                raise ValueError("target_encoder is not fitted. Call fit() first.")
            y = self.target_encoder.transform(df_processed[target_column])
            X = df_processed.drop(columns=[target_column])
            return X, y

        return df_processed

    def fit_transform(self, df, target_column="species", scaling_method="standard"):
        return self.fit(df, target_column, scaling_method).transform(df, target_column)

    def handle_missing_values(self, df, strategy="median"):
        df_clean = df.copy()

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                if strategy == "median":
                    fill_value = df_clean[col].median()
                elif strategy == "mean":
                    fill_value = df_clean[col].mean()
                else:
                    fill_value = 0
                df_clean[col] = df_clean[col].fillna(fill_value)

        categorical_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_value = (
                    df_clean[col].mode()[0]
                    if not df_clean[col].mode().empty
                    else "Unknown"
                )
                df_clean[col] = df_clean[col].fillna(mode_value)

        return df_clean

    def handle_outliers(self, df, method="iqr", threshold=1.5):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        return df_clean

    def preprocess_pipeline(
        self,
        df,
        target_column="species",
        missing_strategy="median",
        outlier_method="iqr",
        scaling_method="standard",
    ):
        self.fit_cleaning_rules(df, target_column, missing_strategy)
        df_clean = self.apply_cleaning_rules(df, target_column)
        X, y = self.fit_transform(df_clean, target_column, scaling_method)
        return X, y, df_clean

    def transform_pipeline(self, df, target_column="species"):
        """按训练集已拟合的清洗、编码和缩放规则转换测试集或验证集。"""
        df_clean = self.apply_cleaning_rules(df, target_column)
        X, y = self.transform(df_clean, target_column)
        return X, y, df_clean

    def save(self, filepath):
        joblib.dump(self, filepath)

    def load(self, filepath):
        loaded_preprocessor = joblib.load(filepath)
        self.label_encoders = loaded_preprocessor.label_encoders
        self.scaler = loaded_preprocessor.scaler
        self.scaling_method = loaded_preprocessor.scaling_method
        self.numeric_features = loaded_preprocessor.numeric_features
        self.categorical_features = loaded_preprocessor.categorical_features
        self.target_encoder = loaded_preprocessor.target_encoder
        self.missing_values = getattr(loaded_preprocessor, "missing_values", {})
        self.outlier_bounds = getattr(loaded_preprocessor, "outlier_bounds", {})
        return self


def preprocess_data(file_path, output_dir="saved_models", target_column="species"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path, encoding="utf-8")

    preprocessor = DataPreprocessor()
    X, y, df_clean = preprocessor.preprocess_pipeline(df, target_column)

    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))

    processed_data_path = os.path.join(output_dir, "processed_data.npz")
    np.savez(processed_data_path, X=X.values, y=y, feature_names=X.columns.tolist())

    print(f"数据预处理完成!")
    print(f"处理后样本数: {len(X)}")
    print(f"特征数: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")

    return X, y, preprocessor


if __name__ == "__main__":
    X, y, preprocessor = preprocess_data("../data/final_stonefly_dataset.csv")
