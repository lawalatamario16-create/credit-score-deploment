"""
pipeline.py
============
Local machine learning training pipeline (OOP) with MLflow tracking
for Credit_Score classification (Dataset A - DTSC6012001 Model Deployment).

Classes
-------
DataPreprocessing : cleaning, feature engineering, and the sklearn
                     preprocessing pipeline (imputation, scaling, encoding).
ModelTraining      : trains a single model, with optional hyperparameter
                     tuning via RandomizedSearchCV.
ModelEvaluation    : computes metrics, classification report, confusion
                     matrix, and feature importance for a fitted model.
TrainingPipeline   : orchestrates preprocessing -> training -> evaluation
                     for every model, logs each run to MLflow, and saves
                     the best model + preprocessing artifacts for deployment.

Usage
-----
    python pipeline.py --data data_A.csv --experiment Credit_Score_Local_Pipeline

Re-running the script (e.g. after the dataset is refreshed) will train and
log a brand-new set of experiments to MLflow without touching earlier runs,
making retraining straightforward, tracked, and monitorable via `mlflow ui`.
"""

import os
import argparse

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it also runs headless / from CLI
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import randint, uniform

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


RANDOM_STATE = 42
TARGET_COL = "Credit_Score"
ID_COLS = ["ID", "Customer_ID", "Name", "SSN", "Month"]
NUMERIC_CLEAN_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
]


# ============================================================
# 1. PREPROCESSING
# ============================================================
class DataPreprocessing:
    """Handle raw data cleaning, feature engineering, and the sklearn
    ColumnTransformer pipeline (imputation + scaling + one-hot encoding).
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.numeric_cols = None
        self.categorical_cols = None

    # ---- cleaning helpers -------------------------------------------------
    @staticmethod
    def _clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
        for col in NUMERIC_CLEAN_COLS:
            df[col] = df[col].astype(str).str.replace("_", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[(df["Age"] < 14) | (df["Age"] > 100), "Age"] = np.nan
        df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
        df.loc[(df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 20), "Num_of_Loan"] = np.nan
        df.loc[df["Interest_Rate"] > 40, "Interest_Rate"] = np.nan
        df.loc[(df["Num_of_Delayed_Payment"] < 0) | (df["Num_of_Delayed_Payment"] > 40), "Num_of_Delayed_Payment"] = np.nan
        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        # Credit_History_Age: "22 Years and 1 Months" -> total bulan
        extracted = df["Credit_History_Age"].str.extract(
            r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?"
        ).astype(float)
        df["Credit_History_Age_Months"] = extracted[0] * 12 + extracted[1]
        df.drop(columns=["Credit_History_Age"], inplace=True)

        def count_loan_types(text):
            if pd.isna(text) or text.strip().lower() == "not specified":
                return 0
            parts = [p.strip() for p in text.replace(" and ", ",").split(",") if p.strip()]
            parts = [p for p in parts if p.lower() != "not specified"]
            return len(parts)

        df["Num_Loan_Types"] = df["Type_of_Loan"].apply(count_loan_types)
        df.drop(columns=["Type_of_Loan"], inplace=True)
        return df

    @staticmethod
    def _clean_placeholders(df: pd.DataFrame) -> pd.DataFrame:
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
        df.drop(columns=ID_COLS, inplace=True)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning + feature-engineering sequence."""
        df = df.copy()
        df = self._clean_numeric_strings(df)
        df = self._fix_outliers(df)
        df = self._feature_engineering(df)
        df = self._clean_placeholders(df)
        return df

    def split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split into train/test and label-encode the target."""
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        y_encoded = self.label_encoder.fit_transform(y)

        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        return train_test_split(
            X, y_encoded, test_size=test_size,
            random_state=self.random_state, stratify=y_encoded,
        )

    def build_pipeline(self) -> ColumnTransformer:
        self.preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), self.numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), self.categorical_cols),
        ])
        return self.preprocessor

    def fit_transform(self, X_train: pd.DataFrame):
        self.build_pipeline()
        return self._to_dense(self.preprocessor.fit_transform(X_train))

    def transform(self, X):
        return self._to_dense(self.preprocessor.transform(X))

    @staticmethod
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else X

    def get_feature_names(self):
        ohe_names = self.preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(
            self.categorical_cols
        )
        return self.numeric_cols + list(ohe_names)


# ============================================================
# 2. TRAINING
# ============================================================
class ModelTraining:
    """Trains a single estimator, optionally tuned with RandomizedSearchCV."""

    def __init__(self, name: str, estimator, param_distributions: dict | None = None,
                 random_state: int = RANDOM_STATE):
        self.name = name
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.tuned = False

    def fit(self, X_train, y_train, tune: bool = False, n_iter: int = 20,
            cv: int = 3, scoring: str = "f1_macro"):
        if tune and self.param_distributions:
            search = RandomizedSearchCV(
                estimator=self.estimator,
                param_distributions=self.param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )
            search.fit(X_train, y_train)
            self.best_estimator_ = search.best_estimator_
            self.best_params_ = search.best_params_
            self.tuned = True
        else:
            self.estimator.fit(X_train, y_train)
            self.best_estimator_ = self.estimator
            self.best_params_ = self.estimator.get_params()
            self.tuned = False
        return self.best_estimator_


# ============================================================
# 3. EVALUATION
# ============================================================
class ModelEvaluation:
    """Computes metrics and generates evaluation artifacts for a fitted model."""

    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }
        return metrics, y_pred

    def classification_report_str(self, y_test, y_pred) -> str:
        return classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)

    def save_confusion_matrix(self, y_test, y_pred, model_name: str, save_path: str) -> str:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def save_feature_importance(self, model, feature_names, model_name: str, save_path: str):
        if not hasattr(model, "feature_importances_"):
            return None
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(10)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
        plt.title(f"Top 10 Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path


# ============================================================
# 4. PIPELINE ORCHESTRATOR
# ============================================================
class TrainingPipeline:
    """Orchestrates preprocessing -> training -> evaluation for every
    candidate model, logging each run to MLflow, then saves the best
    model + preprocessing artifacts needed for deployment (part 1c).
    """

    def __init__(self, data_path: str, experiment_name: str = "Credit_Score_Local_Pipeline",
                 artifacts_dir: str = "artifacts", n_iter: int = 20, cv: int = 3,
                 random_state: int = RANDOM_STATE):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.artifacts_dir = artifacts_dir
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessing = DataPreprocessing(random_state=random_state)
        self.evaluator = None

    def _model_configs(self):
        """Registry of every model to spot-check. Models with a
        `param_distributions` dict are tuned with RandomizedSearchCV;
        the rest are trained once with default parameters."""
        configs = {
            "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=self.random_state), None),
            "KNN": (KNeighborsClassifier(), None),
            "NaiveBayes": (GaussianNB(), None),
            "DecisionTree": (DecisionTreeClassifier(random_state=self.random_state), None),
            "RandomForest": (RandomForestClassifier(random_state=self.random_state), None),
            "GradientBoosting": (GradientBoostingClassifier(random_state=self.random_state), None),
        }
        if XGBClassifier is not None:
            configs["XGBoost"] = (
                XGBClassifier(eval_metric="mlogloss", random_state=self.random_state),
                {
                    "n_estimators": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 6, 8, 10],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "gamma": [0, 0.1, 0.5, 1],
                },
            )
        if LGBMClassifier is not None:
            configs["LightGBM"] = (
                LGBMClassifier(random_state=self.random_state, verbose=-1),
                {
                    "n_estimators": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.2),
                    "num_leaves": randint(20, 150),
                    "max_depth": randint(3, 15),
                    "min_child_samples": randint(10, 100),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                    "reg_alpha": uniform(0, 1),
                    "reg_lambda": uniform(0, 1),
                },
            )
        return configs

    def run(self):
        mlflow.set_experiment(self.experiment_name)

        # ---- 1. Load & preprocess -----------------------------------
        df_raw = pd.read_csv(self.data_path, index_col=0)
        df_clean = self.preprocessing.clean(df_raw)
        X_train, X_test, y_train, y_test = self.preprocessing.split(df_clean)
        X_train_proc = self.preprocessing.fit_transform(X_train)
        X_test_proc = self.preprocessing.transform(X_test)
        feature_names = self.preprocessing.get_feature_names()

        self.evaluator = ModelEvaluation(self.preprocessing.label_encoder)

        # ---- 2. Train + evaluate + log every candidate model --------
        results = []
        run_ids = {}

        for name, (estimator, param_dist) in self._model_configs().items():
            tune = param_dist is not None
            print(f"Training {name} (tune={tune}) ...")

            with mlflow.start_run(run_name=name) as run:
                trainer = ModelTraining(name, estimator, param_dist, random_state=self.random_state)
                fitted_model = trainer.fit(
                    X_train_proc, y_train, tune=tune, n_iter=self.n_iter, cv=self.cv
                )

                metrics, y_pred = self.evaluator.evaluate(fitted_model, X_test_proc, y_test)
                report = self.evaluator.classification_report_str(y_test, y_pred)

                cm_path = os.path.join(self.artifacts_dir, f"confusion_matrix_{name}.png")
                self.evaluator.save_confusion_matrix(y_test, y_pred, name, cm_path)

                report_path = os.path.join(self.artifacts_dir, f"classification_report_{name}.txt")
                with open(report_path, "w") as f:
                    f.write(report)

                fi_path = os.path.join(self.artifacts_dir, f"feature_importance_{name}.png")
                fi_saved = self.evaluator.save_feature_importance(fitted_model, feature_names, name, fi_path)

                # ---- MLflow logging ----
                mlflow.log_param("model_name", name)
                mlflow.log_param("tuned", tune)
                mlflow.log_param("n_train_rows", X_train_proc.shape[0])
                mlflow.log_param("n_features", X_train_proc.shape[1])
                if tune:
                    mlflow.log_params({f"best_{k}": v for k, v in trainer.best_params_.items()})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(report_path)
                if fi_saved:
                    mlflow.log_artifact(fi_saved)
                mlflow.sklearn.log_model(fitted_model, artifact_path="model")

                run_ids[name] = run.info.run_id
                results.append({"model": name, **metrics})

        # ---- 3. Pick the best model across all runs ------------------
        results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
        print("\n=== Ringkasan seluruh eksperimen ===")
        print(results_df)

        best_name = results_df.iloc[0]["model"]
        best_run_id = run_ids[best_name]
        print(f"\nModel terbaik: {best_name} (mlflow run_id={best_run_id})")

        client = mlflow.tracking.MlflowClient()
        client.set_tag(best_run_id, "best_model", "true")

        best_model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.sklearn.load_model(best_model_uri)

        # ---- 4. Save artifacts needed for deployment (part 1c) -------
        joblib.dump(best_model, os.path.join(self.artifacts_dir, "credit_score_model.pkl"))
        joblib.dump(self.preprocessing.preprocessor, os.path.join(self.artifacts_dir, "preprocessor.pkl"))
        joblib.dump(self.preprocessing.label_encoder, os.path.join(self.artifacts_dir, "label_encoder.pkl"))
        joblib.dump(
            {"numeric_cols": self.preprocessing.numeric_cols,
             "categorical_cols": self.preprocessing.categorical_cols},
            os.path.join(self.artifacts_dir, "feature_schema.pkl"),
        )
        results_df.to_csv(os.path.join(self.artifacts_dir, "benchmark_results.csv"), index=False)

        print(f"\nSemua artefak disimpan di folder: {self.artifacts_dir}/")
        return results_df, best_name, best_model


# ============================================================
# CLI ENTRY POINT
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Local ML training pipeline with MLflow tracking.")
    parser.add_argument("--data", type=str, default="data_A.csv", help="Path ke file CSV data training.")
    parser.add_argument("--experiment", type=str, default="Credit_Score_Local_Pipeline",
                         help="Nama MLflow experiment.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                         help="Folder untuk menyimpan model, plot, dan hasil evaluasi.")
    parser.add_argument("--n-iter", type=int, default=20,
                         help="Jumlah kombinasi hyperparameter yang dicoba RandomizedSearchCV.")
    parser.add_argument("--cv", type=int, default=3, help="Jumlah fold cross-validation saat tuning.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrainingPipeline(
        data_path=args.data,
        experiment_name=args.experiment,
        artifacts_dir=args.artifacts_dir,
        n_iter=args.n_iter,
        cv=args.cv,
    )
    pipeline.run()
