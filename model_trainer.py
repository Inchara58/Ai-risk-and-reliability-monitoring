"""
AEGIS AI — Master Model Factory & Predictive Intelligence Engine
File: model_trainer.py
Description: Enterprise-grade model training, multi-algorithm benchmarking, cross-validation,
             feature importance extraction, artifact serialization, and real-time/batch inference
             for both Classification and Regression problems.
"""

import os
import json
import time
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# DATASET ANALYSIS & PROBLEM DETECTION HELPERS
# ==============================================================================

def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Identifies obvious primary keys or ID columns that should be excluded from ML training."""
    id_cols = []
    n_rows = len(df)
    for col in df.columns:
        col_lower = str(col).lower()
        if (
            col_lower.endswith("_id") or
            col_lower.startswith("id_") or
            col_lower in ["id", "uuid", "guid", "account_id", "machine_id", "order_id",
                          "customer_id", "patient_id", "shipment_id", "transaction_id"]
        ):
            id_cols.append(col)
        elif df[col].dtype == "object" and df[col].nunique(dropna=True) > (n_rows * 0.95) and n_rows > 50:
            id_cols.append(col)
    return id_cols


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Dynamically determines if target is Classification or Regression.
    Never assumes - returns 'classification' or 'regression'.
    """
    if target_col not in df.columns:
        return "classification"

    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()

    # If boolean or object / string or categorical
    if target_series.dtype == "object" or target_series.dtype == "bool" or target_series.dtype.name == "category":
        return "classification"

    # If numeric but only 2 to 10 unique values, or binary 0/1
    if set(target_series.unique()).issubset({0, 1, 0.0, 1.0}):
        return "classification"

    if n_unique <= 10:
        return "classification"

    # Continuous float/integer target with large cardinality
    return "regression"


def analyze_dataset_features(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Profiles features for the Model Factory configuration UI:
    Detects data types, missing rates, unique counts, distributions, and ID candidates.
    """
    total_rows, total_cols = df.shape
    id_candidates = detect_id_columns(df)
    
    # Auto-detect default target if none provided
    suggested_target = target_col
    if not suggested_target or suggested_target not in df.columns:
        possible_targets = [
            "Machine_Failure", "Returned", "Default", "Readmitted", "Delayed",
            "Loan_Approved", "Churn", "Target", "Status", "Delivery_Days", "Fuel_Cost"
        ]
        for pt in possible_targets:
            if pt in df.columns:
                suggested_target = pt
                break
        if not suggested_target and len(df.columns) > 0:
            suggested_target = df.columns[-1]

    problem_type = detect_problem_type(df, suggested_target) if suggested_target else "classification"

    features_summary = []
    num_cols = []
    cat_cols = []

    for col in df.columns:
        is_target = (col == suggested_target)
        is_id = col in id_candidates
        dtype_str = str(df[col].dtype)
        is_numeric = np.issubdtype(df[col].dtype, np.number)
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        unique_cnt = int(df[col].nunique(dropna=True))

        info = {
            "name": col,
            "type": "Numeric" if is_numeric else "Categorical",
            "dtype": dtype_str,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_cnt,
            "is_target": is_target,
            "is_id_candidate": is_id,
            "default_selected": not is_target and not is_id
        }

        if is_numeric:
            valid = df[col].dropna()
            if len(valid) > 0:
                info.update({
                    "min": round(float(valid.min()), 2),
                    "max": round(float(valid.max()), 2),
                    "mean": round(float(valid.mean()), 2),
                    "median": round(float(valid.median()), 2)
                })
            num_cols.append(col)
        else:
            cat_cols.append(col)
            # Sample categories
            unique_vals = [str(x) for x in df[col].dropna().unique()[:20]]
            info["categories_sample"] = unique_vals

        features_summary.append(info)

    feature_details = {info["name"]: info for info in features_summary}
    problem_types = {col: detect_problem_type(df, col) for col in df.columns}

    return {
        "columns": list(df.columns),
        "total_rows": total_rows,
        "total_cols": total_cols,
        "suggested_target": suggested_target,
        "recommended_target": suggested_target,
        "detected_problem_type": problem_type,
        "problem_types": problem_types,
        "id_columns": id_candidates,
        "numerical_columns": [c for c in num_cols if c != suggested_target],
        "categorical_columns": [c for c in cat_cols if c != suggested_target],
        "features": features_summary,
        "feature_details": feature_details
    }


# ==============================================================================
# MASTER MODEL TRAINER & EVALUATION ENGINE
# ==============================================================================

class ModelTrainer:
    """
    Production-grade Model Training Pipeline:
    - Preprocessing with missing imputation and dummy/category encoding
    - Multi-algorithm training (Logistic/Ridge, Random Forest, Gradient Boosting)
    - True metric evaluation (Accuracy, Precision, Recall, F1, ROC-AUC / MAE, MSE, RMSE, R²)
    - 5-Fold Cross Validation
    - Feature importance calculation
    - Full artifact packaging and persistence
    """

    SUPPORTED_ALGORITHMS = {
        "classification": [
            {"id": "random_forest", "name": "Random Forest Classifier", "category": "Ensemble Trees", "desc": "100 bagged decision trees with out-of-bag error estimation."},
            {"id": "gradient_boosting", "name": "Gradient Boosting Classifier", "category": "Sequential Boosting", "desc": "Iteratively fits weak learners to gradient loss residuals."},
            {"id": "logistic_regression", "name": "Logistic Regression", "category": "Linear Model", "desc": "L2-regularized linear decision boundary with scaled covariates."}
        ],
        "regression": [
            {"id": "random_forest_regressor", "name": "Random Forest Regressor", "category": "Ensemble Trees", "desc": "Averages variance across 100 deep regression trees."},
            {"id": "gradient_boosting_regressor", "name": "Gradient Boosting Regressor", "category": "Sequential Boosting", "desc": "Boosted regression trees optimizing least squares loss."},
            {"id": "ridge_regression", "name": "Ridge Linear Regressor", "category": "Linear Model", "desc": "Standardized linear model with L2 Tikhonov regularization."}
        ]
    }

    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 problem_type: Optional[str] = None,
                 selected_features: Optional[List[str]] = None,
                 excluded_features: Optional[List[str]] = None,
                 random_state: int = 42):
        self.df = df.copy()
        self.target_col = target_col
        self.random_state = random_state

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' does not exist in dataset.")

        # Determine problem type
        self.problem_type = problem_type or detect_problem_type(self.df, self.target_col)

        # Feature selection setup
        all_feature_candidates = [c for c in self.df.columns if c != self.target_col]
        auto_id_cols = detect_id_columns(self.df)

        if selected_features:
            self.selected_features = [f for f in selected_features if f in all_feature_candidates]
        else:
            self.selected_features = [f for f in all_feature_candidates if f not in auto_id_cols]

        if excluded_features:
            self.selected_features = [f for f in self.selected_features if f not in excluded_features]

        if not self.selected_features:
            raise ValueError("No features available for training after exclusions.")

        # Preprocessing metadata containers
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_imputes = {}
        self.categorical_imputes = {}
        self.categorical_dummies_map = {}
        self.scaler = None
        self.encoded_feature_names = []
        self.target_classes = []
        self.target_mean = None

    def _fit_preprocessor(self, X_df: pd.DataFrame, y_series: pd.Series):
        """Fits imputer stats, category levels, and scalers on training data."""
        self.numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Learn numeric medians
        for col in self.numeric_cols:
            med = float(X_df[col].median()) if not np.isnan(X_df[col].median()) else 0.0
            self.numeric_imputes[col] = med

        # Learn categorical modes and distinct values
        for col in self.categorical_cols:
            mode_val = X_df[col].mode()
            self.categorical_imputes[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "Unknown"
            cats = [str(x) for x in X_df[col].dropna().unique() if str(x).strip() != ""]
            self.categorical_dummies_map[col] = cats

        if self.problem_type == "classification":
            self.target_classes = [str(c) for c in np.unique(y_series.dropna())]
        else:
            self.target_mean = float(y_series.dropna().mean())

    def _transform_dataframe(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted preprocessor to raw feature dataframe."""
        X_out = pd.DataFrame(index=X_df.index)

        # Numeric transformation
        for col in self.numeric_cols:
            val_series = pd.to_numeric(X_df[col], errors='coerce') if col in X_df.columns else pd.Series(self.numeric_imputes.get(col, 0.0), index=X_df.index)
            impute_val = self.numeric_imputes.get(col, 0.0)
            X_out[col] = val_series.fillna(impute_val).astype(float)

        # Categorical transformation (One-Hot Encoding with exact known categories)
        for col in self.categorical_cols:
            if col in X_df.columns:
                raw_str = X_df[col].fillna(self.categorical_imputes.get(col, "Unknown")).astype(str).str.strip()
            else:
                raw_str = pd.Series(self.categorical_imputes.get(col, "Unknown"), index=X_df.index)

            known_cats = self.categorical_dummies_map.get(col, [])
            for cat in known_cats:
                dummy_col_name = f"{col}_{cat}"
                X_out[dummy_col_name] = (raw_str == cat).astype(float)

        return X_out

    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepares full train/test split arrays."""
        # Drop rows where target is null
        valid_df = self.df.dropna(subset=[self.target_col]).copy()
        if len(valid_df) < 10:
            raise ValueError(f"Dataset has only {len(valid_df)} rows with valid target values. Minimum 10 rows required.")

        X_raw = valid_df[self.selected_features].copy()
        y_raw = valid_df[self.target_col].copy()

        # Target encoding
        if self.problem_type == "classification":
            if y_raw.dtype == "object" or y_raw.dtype == "bool" or y_raw.dtype.name == "category":
                # Convert binary/multiclass strings to ints
                classes, y_encoded = np.unique(y_raw.astype(str), return_inverse=True)
                self.target_classes = list(classes)
                y = y_encoded
            else:
                # Numeric binary or discrete
                y = y_raw.astype(int).values
                self.target_classes = [str(c) for c in np.unique(y)]
        else:
            y = pd.to_numeric(y_raw, errors='coerce').fillna(y_raw.median()).astype(float).values

        # Split 80/20 train/test
        is_stratified = (self.problem_type == "classification" and len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.20, random_state=self.random_state,
            stratify=y if is_stratified else None
        )

        # Fit preprocessor strictly on training fold
        self._fit_preprocessor(X_train_raw, pd.Series(y_train))

        # Transform both folds
        X_train_df = self._transform_dataframe(X_train_raw)
        X_test_df = self._transform_dataframe(X_test_raw)
        self.encoded_feature_names = list(X_train_df.columns)

        return X_train_df.values, X_test_df.values, y_train, y_test, X_test_df

    def train_single_model(self, algorithm_id: str) -> Dict[str, Any]:
        """Trains one specific algorithm and computes all governance and performance metrics."""
        t_start = time.time()
        X_train, X_test, y_train, y_test, X_test_df = self.prepare_dataset()

        # Instantiate algorithm
        model = None
        algo_name = algorithm_id
        is_linear = False

        if self.problem_type == "classification":
            if algorithm_id == "random_forest":
                algo_name = "Random Forest Classifier"
                model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, random_state=self.random_state, n_jobs=-1)
            elif algorithm_id == "gradient_boosting":
                algo_name = "Gradient Boosting Classifier"
                model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=self.random_state)
            else:
                algo_name = "Logistic Regression"
                is_linear = True
                model = LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state)
        else:
            # Regression
            if algorithm_id == "random_forest_regressor":
                algo_name = "Random Forest Regressor"
                model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=self.random_state, n_jobs=-1)
            elif algorithm_id == "gradient_boosting_regressor":
                algo_name = "Gradient Boosting Regressor"
                model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=self.random_state)
            else:
                algo_name = "Ridge Linear Regressor"
                is_linear = True
                model = Ridge(alpha=1.0, random_state=self.random_state)

        # Scale data for linear models
        scaler = None
        if is_linear:
            scaler = StandardScaler()
            X_train_fit = scaler.fit_transform(X_train)
            X_test_fit = scaler.transform(X_test)
        else:
            X_train_fit = X_train
            X_test_fit = X_test

        # Fit model
        model.fit(X_train_fit, y_train)
        train_time_sec = round(time.time() - t_start, 3)

        # Predictions
        y_pred = model.predict(X_test_fit)

        # Metric evaluations
        metrics = {
            "algorithm_id": algorithm_id,
            "algorithm_name": algo_name,
            "problem_type": self.problem_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time_sec": train_time_sec,
            "features_count": len(self.selected_features),
            "encoded_features_count": len(self.encoded_feature_names)
        }

        # 5-Fold Cross Validation
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) if self.problem_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scoring_metric = "f1_weighted" if self.problem_type == "classification" else "r2"
        cv_scores = cross_val_score(model, X_train_fit, y_train, cv=cv_splitter, scoring=scoring_metric)
        metrics["cv_mean"] = round(float(np.mean(cv_scores)), 4)
        metrics["cv_std"] = round(float(np.std(cv_scores)), 4)

        if self.problem_type == "classification":
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            # ROC-AUC calculation
            auc_val = 0.0
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_fit)
                    if len(self.target_classes) == 2:
                        auc_val = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        auc_val = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
                except Exception:
                    auc_val = 0.5

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                "matrix": cm.tolist(),
                "labels": self.target_classes
            }

            metrics.update({
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc_val, 4),
                "confusion_matrix": cm_dict,
                "primary_metric_name": "F1-Score",
                "primary_metric_val": round(f1, 4)
            })
        else:
            # Regression Metrics
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            metrics.update({
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "r2_score": round(r2, 4),
                "primary_metric_name": "R² Variance",
                "primary_metric_val": round(r2, 4)
            })

        # Feature Importances Extraction
        raw_importances = {}
        if hasattr(model, "feature_importances_"):
            for fname, imp in zip(self.encoded_feature_names, model.feature_importances_):
                raw_importances[fname] = float(imp)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).flatten() if hasattr(model.coef_, 'flatten') else np.abs(model.coef_)
            for fname, coef in zip(self.encoded_feature_names, coefs):
                raw_importances[fname] = float(coef)
        else:
            for fname in self.encoded_feature_names:
                raw_importances[fname] = 1.0 / len(self.encoded_feature_names)

        # Aggregate dummy importances back to parent selected features
        parent_importances = {f: 0.0 for f in self.selected_features}
        for enc_f, imp_val in raw_importances.items():
            for raw_f in self.selected_features:
                if enc_f == raw_f or enc_f.startswith(f"{raw_f}_"):
                    parent_importances[raw_f] += imp_val
                    break

        total_imp = sum(parent_importances.values()) or 1.0
        feature_importance_list = []
        for feat, raw_val in sorted(parent_importances.items(), key=lambda x: x[1], reverse=True):
            pct = round((raw_val / total_imp) * 100, 2)
            feature_importance_list.append({
                "feature": feat,
                "importance_score": round(raw_val, 4),
                "importance_pct": pct
            })

        metrics["feature_importances"] = feature_importance_list

        # Package trained model bundle
        bundle = {
            "model": model,
            "scaler": scaler,
            "is_linear": is_linear,
            "problem_type": self.problem_type,
            "target_col": self.target_col,
            "target_classes": self.target_classes,
            "target_mean": self.target_mean,
            "selected_features": self.selected_features,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_imputes": self.numeric_imputes,
            "categorical_imputes": self.categorical_imputes,
            "categorical_dummies_map": self.categorical_dummies_map,
            "encoded_feature_names": self.encoded_feature_names,
            "metrics": metrics
        }

        return {
            "metrics": metrics,
            "bundle": bundle
        }

    def compare_all_models(self) -> Dict[str, Any]:
        """
        Trains and benchmarks all 3 supported algorithms for the problem type.
        Returns full comparative leaderboard and identifies the recommended best model with clear criteria.
        """
        algos = self.SUPPORTED_ALGORITHMS[self.problem_type]
        results_list = []
        bundles_dict = {}

        for algo in algos:
            res = self.train_single_model(algo["id"])
            m = res["metrics"]
            m["category"] = algo["category"]
            m["desc"] = algo["desc"]
            results_list.append(m)
            bundles_dict[algo["id"]] = res["bundle"]

        # Determine Best Model
        if self.problem_type == "classification":
            # Highest F1 Score, broken by ROC-AUC and CV stability
            best_m = max(results_list, key=lambda x: (x.get("f1_score", 0), x.get("roc_auc", 0), x.get("cv_mean", 0)))
            recommendation_reason = (
                f"Selected as top performing model with highest F1-Score ({best_m['f1_score']:.4f}) "
                f"and robust 5-Fold Cross-Validation ({best_m['cv_mean']:.4f} ± {best_m['cv_std']:.4f})."
            )
        else:
            # Highest R2 Score, broken by lowest RMSE
            best_m = max(results_list, key=lambda x: (x.get("r2_score", -999), -x.get("rmse", 999)))
            recommendation_reason = (
                f"Selected as top performing model with highest R² Variance ({best_m['r2_score']:.4f}) "
                f"and lowest Root Mean Squared Error ({best_m['rmse']:.4f})."
            )

        for m in results_list:
            m["is_recommended"] = (m["algorithm_id"] == best_m["algorithm_id"])

        return {
            "problem_type": self.problem_type,
            "target_column": self.target_col,
            "selected_features": self.selected_features,
            "leaderboard": results_list,
            "recommended_model_id": best_m["algorithm_id"],
            "recommended_model_name": best_m["algorithm_name"],
            "recommendation_reason": recommendation_reason,
            "bundles": bundles_dict
        }


# ==============================================================================
# MODEL ARTIFACT SERIALIZATION & PERSISTENCE
# ==============================================================================

def save_model_artifact(project_id: str, model_id: str, bundle: Dict[str, Any]) -> str:
    """Saves trained model bundle to compressed joblib file."""
    proj_dir = os.path.join(MODELS_DIR, str(project_id))
    os.makedirs(proj_dir, exist_ok=True)
    file_path = os.path.join(proj_dir, f"{model_id}.joblib")
    joblib.dump(bundle, file_path, compress=3)
    return file_path


def load_model_artifact(model_path: str) -> Optional[Dict[str, Any]]:
    """Loads model bundle from disk."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model artifact from {model_path}: {e}")
        return None


# ==============================================================================
# PREDICTION STUDIO REAL-TIME & BATCH INFERENCE ENGINE
# ==============================================================================

class ModelPredictor:
    """
    Executes single-record and batch predictions using a trained model artifact.
    Includes feature transformation, probability estimation, and feature attribution explanations.
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.is_linear = bundle.get("is_linear", False)
        self.problem_type = bundle["problem_type"]
        self.target_col = bundle["target_col"]
        self.target_classes = bundle.get("target_classes", [])
        self.target_mean = bundle.get("target_mean")
        self.selected_features = bundle["selected_features"]
        self.numeric_cols = bundle["numeric_cols"]
        self.categorical_cols = bundle["categorical_cols"]
        self.numeric_imputes = bundle["numeric_imputes"]
        self.categorical_imputes = bundle["categorical_imputes"]
        self.categorical_dummies_map = bundle["categorical_dummies_map"]
        self.encoded_feature_names = bundle["encoded_feature_names"]
        self.metrics = bundle.get("metrics", {})

    def _transform_inputs(self, input_df: pd.DataFrame) -> np.ndarray:
        """Transforms raw input rows into the exact model feature vector."""
        X_out = pd.DataFrame(index=input_df.index)

        # Numeric transformations
        for col in self.numeric_cols:
            if col in input_df.columns:
                val_s = pd.to_numeric(input_df[col], errors='coerce')
            else:
                val_s = pd.Series(self.numeric_imputes.get(col, 0.0), index=input_df.index)
            impute_val = self.numeric_imputes.get(col, 0.0)
            X_out[col] = val_s.fillna(impute_val).astype(float)

        # Categorical transformations
        for col in self.categorical_cols:
            if col in input_df.columns:
                raw_s = input_df[col].fillna(self.categorical_imputes.get(col, "Unknown")).astype(str).str.strip()
            else:
                raw_s = pd.Series(self.categorical_imputes.get(col, "Unknown"), index=input_df.index)

            known_cats = self.categorical_dummies_map.get(col, [])
            for cat in known_cats:
                dummy_name = f"{col}_{cat}"
                X_out[dummy_name] = (raw_s == cat).astype(float)

        # Ensure exact column order
        for col in self.encoded_feature_names:
            if col not in X_out.columns:
                X_out[col] = 0.0
        X_out = X_out[self.encoded_feature_names]

        if self.is_linear and self.scaler:
            return self.scaler.transform(X_out.values)
        return X_out.values

    def predict_single(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs inference on a single user-provided record.
        Returns prediction, probability/confidence breakdown, and local feature attribution.
        """
        t0 = time.time()
        input_df = pd.DataFrame([raw_input])
        X_vec = self._transform_inputs(input_df)

        raw_pred = self.model.predict(X_vec)[0]
        latency_ms = round((time.time() - t0) * 1000, 2)

        out = {
            "problem_type": self.problem_type,
            "target_column": self.target_col,
            "latency_ms": latency_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_summary": raw_input
        }

        if self.problem_type == "classification":
            # Map index to class label if applicable
            pred_class_label = str(raw_pred)
            if self.target_classes and int(raw_pred) < len(self.target_classes):
                pred_class_label = self.target_classes[int(raw_pred)]

            confidence_pct = 95.0
            probabilities = {}
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_vec)[0]
                for idx, p in enumerate(probs):
                    cls_name = self.target_classes[idx] if idx < len(self.target_classes) else f"Class {idx}"
                    probabilities[str(cls_name)] = round(float(p) * 100, 2)
                confidence_pct = max(probabilities.values()) if probabilities else 95.0

            out.update({
                "prediction": pred_class_label,
                "prediction_raw": int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else str(raw_pred),
                "confidence_pct": confidence_pct,
                "class_probabilities": probabilities
            })
        else:
            # Regression Prediction
            pred_val = float(raw_pred)
            out.update({
                "prediction": round(pred_val, 2),
                "prediction_raw": pred_val,
                "unit": self.target_col
            })

        # Feature Attribution Explanations
        attributions = []
        importances = {item["feature"]: item["importance_pct"] for item in self.metrics.get("feature_importances", [])}
        for feat in self.selected_features:
            user_val = raw_input.get(feat)
            imp = importances.get(feat, 5.0)
            if feat in self.numeric_cols:
                baseline = self.numeric_imputes.get(feat, 0.0)
                val_num = float(user_val) if user_val is not None and str(user_val) != "" else baseline
                delta = val_num - baseline
                direction = "Positive Influence" if delta >= 0 else "Negative Influence"
                attributions.append({
                    "feature": feat,
                    "value": val_num,
                    "baseline": baseline,
                    "importance_pct": imp,
                    "contribution": direction
                })
            else:
                attributions.append({
                    "feature": feat,
                    "value": str(user_val) if user_val is not None else "Unknown",
                    "importance_pct": imp,
                    "contribution": "Observed Factor"
                })

        out["feature_attributions"] = sorted(attributions, key=lambda x: x["importance_pct"], reverse=True)[:6]
        out["local_attributions"] = out["feature_attributions"]
        out["probabilities"] = out.get("class_probabilities", {})
        return out

    def predict_batch(self, batch_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes high-throughput batch inference on a dataset of records.
        Returns predicted dataset preview, summary distribution, and validation logs.
        """
        t0 = time.time()
        n_rows = len(batch_df)
        if n_rows == 0:
            raise ValueError("Provided batch dataset is empty.")

        # Validate features
        missing_features = [f for f in self.selected_features if f not in batch_df.columns]
        warnings = []
        if missing_features:
            warnings.append(f"Missing {len(missing_features)} feature(s) from input: {', '.join(missing_features[:5])}. Imputed with training defaults.")

        X_mat = self._transform_inputs(batch_df)
        raw_preds = self.model.predict(X_mat)
        latency_ms = round((time.time() - t0) * 1000, 2)

        result_df = batch_df.copy()

        if self.problem_type == "classification":
            # Map predictions to class labels
            pred_labels = []
            for p in raw_preds:
                if self.target_classes and int(p) < len(self.target_classes):
                    pred_labels.append(self.target_classes[int(p)])
                else:
                    pred_labels.append(str(p))
            result_df["AEGIS_Prediction"] = pred_labels

            # Probabilities if supported
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_mat)
                max_probs = np.max(probs, axis=1) * 100
                result_df["AEGIS_Confidence_Pct"] = np.round(max_probs, 1)

            # Class Distribution summary
            dist = result_df["AEGIS_Prediction"].value_counts().to_dict()
            distribution_summary = {str(k): int(v) for k, v in dist.items()}
        else:
            result_df["AEGIS_Prediction"] = np.round(raw_preds, 2)
            distribution_summary = {
                "mean_prediction": round(float(np.mean(raw_preds)), 2),
                "std_prediction": round(float(np.std(raw_preds)), 2),
                "min_prediction": round(float(np.min(raw_preds)), 2),
                "max_prediction": round(float(np.max(raw_preds)), 2)
            }

        # Provide a clean, serializable sample (first 100 rows)
        sample_records = result_df.head(100).replace({np.nan: None}).to_dict(orient="records")

        return {
            "total_records": n_rows,
            "total_records_processed": n_rows,
            "latency_ms": latency_ms,
            "problem_type": self.problem_type,
            "target_column": self.target_col,
            "distribution_summary": distribution_summary,
            "warnings": warnings,
            "columns": list(result_df.columns),
            "preview_sample": sample_records,
            "scored_records": sample_records
        }
