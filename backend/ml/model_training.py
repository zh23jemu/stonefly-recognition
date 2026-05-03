import pandas as pd
import numpy as np
import time
import joblib
import os
import json
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, output_dir="saved_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.results = {}

    def prepare_data(
        self,
        X,
        y,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
        max_train_samples=20000,
    ):
        """按训练/测试/验证三份划分数据，并可限制最终训练样本量。

        测试集用于模型评估，验证集用于前端演示选择样本。训练样本上限只
        作用于训练集，不会裁剪测试集和验证集，避免页面演示样本不足。
        """
        print(f"  Using {len(y)} samples for training")

        y_array = np.asarray(y)
        holdout_size = test_size + validation_size
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout_size, random_state=random_state
        )
        train_idx, holdout_idx = next(splitter.split(np.zeros(len(y_array)), y_array))
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_holdout = X.iloc[holdout_idx] if hasattr(X, "iloc") else X[holdout_idx]
        y_train = y_array[train_idx]
        y_holdout = y_array[holdout_idx]

        validation_ratio_in_holdout = validation_size / holdout_size
        holdout_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_ratio_in_holdout,
            random_state=random_state,
        )
        test_rel_idx, validation_rel_idx = next(
            holdout_splitter.split(np.zeros(len(y_holdout)), y_holdout)
        )
        X_test = (
            X_holdout.iloc[test_rel_idx]
            if hasattr(X_holdout, "iloc")
            else X_holdout[test_rel_idx]
        )
        X_validation = (
            X_holdout.iloc[validation_rel_idx]
            if hasattr(X_holdout, "iloc")
            else X_holdout[validation_rel_idx]
        )
        y_test = y_holdout[test_rel_idx]
        y_validation = y_holdout[validation_rel_idx]

        if max_train_samples and len(X_train) > max_train_samples:
            print(f"  Limiting train split to {max_train_samples} samples")
            train_splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=max_train_samples, random_state=random_state
            )
            sampled_idx, _ = next(
                train_splitter.split(np.zeros(len(y_train)), np.asarray(y_train))
            )
            X_train = (
                X_train.iloc[sampled_idx]
                if hasattr(X_train, "iloc")
                else X_train[sampled_idx]
            )
            y_train = np.asarray(y_train)[sampled_idx]

        return X_train, X_test, X_validation, y_train, y_test, y_validation

    def _get_cv_splitter(self, cv, y_train, random_state=42):
        if isinstance(cv, int):
            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        return cv

    def train_random_forest(
        self, X_train, y_train, param_grid=None, cv=3, sample_weight=None
    ):
        print("\nTraining Random Forest (RandomizedSearchCV)...")
        start_time = time.time()

        from scipy.stats import randint

        param_distributions = {
            "n_estimators": randint(50, 200),
            "max_depth": [10, 20, 30, None],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 5),
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=2, class_weight="balanced")
        random_search = RandomizedSearchCV(
            rf,
            param_distributions,
            n_iter=10,
            cv=self._get_cv_splitter(cv, y_train),
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
            random_state=42,
        )
        random_search.fit(X_train, y_train, sample_weight=sample_weight)

        training_time = time.time() - start_time

        self.models["random_forest"] = random_search.best_estimator_
        self.results["random_forest"] = {
            "best_params": random_search.best_params_,
            "best_cv_score": float(random_search.best_score_),
            "training_time": training_time,
        }

        print(
            f"[OK] Random Forest training complete. Best CV score: {random_search.best_score_:.4f}"
        )
        print(f"  Best params: {random_search.best_params_}")

        return random_search.best_estimator_

    def train_svm(self, X_train, y_train, param_grid=None, cv=3, sample_weight=None):
        print("\nTraining SVM...")
        start_time = time.time()

        if param_grid is None:
            param_grid = {
                "C": [1, 10],
                "kernel": ["rbf"],
                "gamma": ["scale"],
            }

        svm = SVC(random_state=42, probability=True, class_weight="balanced")
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=self._get_cv_splitter(cv, y_train),
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models["svm"] = grid_search.best_estimator_
        self.results["svm"] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "training_time": training_time,
        }

        print(
            f"[OK] SVM training complete. Best CV score: {grid_search.best_score_:.4f}"
        )
        print(f"  Best params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def train_xgboost(
        self, X_train, y_train, param_grid=None, cv=3, sample_weight=None
    ):
        print("\nTraining XGBoost...")
        start_time = time.time()

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train_xgb = le.fit_transform(y_train)

        self.xgb_label_encoder = le

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.1],
            }

        xgb = XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=2,
            num_class=len(le.classes_),
        )
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=self._get_cv_splitter(cv, y_train_xgb),
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train_xgb, sample_weight=sample_weight)

        training_time = time.time() - start_time

        self.models["xgboost"] = grid_search.best_estimator_
        self.results["xgboost"] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "training_time": training_time,
            "label_encoder_classes": le.classes_.tolist(),
        }

        print(
            f"[OK] XGBoost training complete. Best CV score: {grid_search.best_score_:.4f}"
        )
        print(f"  Best params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def train_knn(self, X_train, y_train, param_grid=None, cv=3, sample_weight=None):
        print("\nTraining KNN...")
        start_time = time.time()

        if param_grid is None:
            param_grid = {
                "n_neighbors": [5, 10],
                "weights": ["uniform"],
            }

        knn = KNeighborsClassifier(n_jobs=2)
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=self._get_cv_splitter(cv, y_train),
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        training_time = time.time() - start_time

        self.models["knn"] = grid_search.best_estimator_
        self.results["knn"] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "training_time": training_time,
        }

        print(
            f"[OK] KNN training complete. Best CV score: {grid_search.best_score_:.4f}"
        )
        print(f"  Best params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def train_all_models(self, X_train, y_train, cv=3):
        print("\n" + "=" * 50)
        print("Starting model training...")
        print("=" * 50)

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        self.train_random_forest(X_train, y_train, cv=cv, sample_weight=sample_weight)
        self.train_svm(X_train, y_train, cv=cv, sample_weight=sample_weight)
        self.train_xgboost(X_train, y_train, cv=cv, sample_weight=sample_weight)
        self.train_knn(X_train, y_train, cv=cv)

        self.save_results()

        print("\n" + "=" * 50)
        print("All models trained successfully!")
        print("=" * 50)

        return self.models, self.results

    def train_two_stage_models(
        self, X_train, y_train, unknown_class_index, known_threshold=0.4
    ):
        print("\nTraining Two-Stage Models...")

        y_train = np.asarray(y_train)
        y_binary = (y_train != unknown_class_index).astype(int)

        stage1_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=2,
            class_weight="balanced",
        )
        stage1_sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_binary
        )
        stage1_model.fit(X_train, y_binary, sample_weight=stage1_sample_weight)

        known_mask = y_train != unknown_class_index
        X_known = (
            X_train.iloc[known_mask]
            if hasattr(X_train, "iloc")
            else X_train[known_mask]
        )
        y_known = y_train[known_mask]

        stage2_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=30,
            random_state=42,
            n_jobs=2,
            class_weight="balanced_subsample",
        )
        stage2_sample_weight = compute_sample_weight(class_weight="balanced", y=y_known)
        stage2_model.fit(X_known, y_known, sample_weight=stage2_sample_weight)

        self.models["stage1_known_unknown"] = stage1_model
        self.models["stage2_known_species"] = stage2_model

        self.results["two_stage"] = {
            "unknown_class_index": int(unknown_class_index),
            "known_threshold": float(known_threshold),
            "stage1_positive_rate": float(np.mean(y_binary)),
            "stage2_num_known_classes": int(len(np.unique(y_known))),
        }

        print("[OK] Two-stage models trained")
        print(
            f"  Stage1: unknown-vs-known, Stage2: {len(np.unique(y_known))} known classes"
        )

        return stage1_model, stage2_model

    def save_results(self):
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def save_models(self):
        for name, model in self.models.items():
            model_path = os.path.join(self.output_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"[OK] Saved {name} model")


def train_models(X, y, output_dir="saved_models", cv=3):
    trainer = ModelTrainer(output_dir)
    X_train, X_test, X_validation, y_train, y_test, y_validation = trainer.prepare_data(
        X, y
    )

    models, results = trainer.train_all_models(X_train, y_train, cv=cv)
    trainer.save_models()

    np.savez(os.path.join(output_dir, "test_data.npz"), X_test=X_test, y_test=y_test)

    return trainer, X_test, X_validation, y_test, y_validation


if __name__ == "__main__":
    from preprocessing import preprocess_data

    X, y, preprocessor = preprocess_data("../data/final_stonefly_dataset.csv")
    trainer, X_test, y_test = train_models(X, y, cv=3)
