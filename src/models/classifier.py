r"""
GlucoTwin — Hypoglycemia ML Classifier
src/models/classifier.py

Unified model wrapper for:
- LightGBM (Primary recommended tree model)
- XGBoost (Extreme gradient boosting)
- Random Forest (Bagging ensemble baseline)
- Logistic Regression (Linear calibrated baseline)

Features:
- Handles missing value imputation strictly fit on train set
- Native class imbalance handling
- Clinically tuned probability output
- Model persistence (save / load)
"""

from __future__ import annotations

import os
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class HypoClassifier(BaseEstimator, ClassifierMixin):
    """
    Hypoglycemia prediction classifier with integrated preprocessing and imbalance calibration.
    """
    def __init__(
        self,
        model_type: str = "lightgbm",
        random_seed: int = 42,
        class_weight_strategy: str = "balanced",
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.class_weight_strategy = class_weight_strategy
        self.params = dict(params or {})
        if kwargs:
            self.params.update(kwargs)
        
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.model_: Any = None
        self.is_fitted_: bool = False

    def _init_estimator(self, scale_pos_weight: float = 1.0):
        if self.model_type == "lightgbm":
            default_params = {
                "n_estimators": 250,
                "max_depth": 6,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            if self.class_weight_strategy == "balanced":
                default_params["scale_pos_weight"] = scale_pos_weight
            default_params.update(self.params)
            return lgb.LGBMClassifier(**default_params)

        elif self.model_type == "xgboost":
            default_params = {
                "n_estimators": 250,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_seed,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            if self.class_weight_strategy == "balanced":
                default_params["scale_pos_weight"] = scale_pos_weight
            default_params.update(self.params)
            return xgb.XGBClassifier(**default_params)

        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 4,
                "class_weight": "balanced" if self.class_weight_strategy == "balanced" else None,
                "random_state": self.random_seed,
                "n_jobs": -1,
            }
            default_params.update(self.params)
            return RandomForestClassifier(**default_params)

        elif self.model_type == "logistic_regression":
            default_params = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced" if self.class_weight_strategy == "balanced" else None,
                "random_state": self.random_seed,
                "solver": "lbfgs",
            }
            default_params.update(self.params)
            return LogisticRegression(**default_params)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose from: lightgbm, xgboost, random_forest, logistic_regression")

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> HypoClassifier:
        """
        Fit imputer, scaler (if LR), and classifier on training data.
        """
        self.feature_names_ = list(X.columns)
        y_arr = np.asarray(y).astype(int)
        
        # Fit imputer strictly on X_train (median strategy is robust to physiological outliers)
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)
        
        # Calculate imbalance weight
        n_pos = np.sum(y_arr == 1)
        n_neg = np.sum(y_arr == 0)
        scale_pos_weight = float(n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        
        # Scaler for linear models
        if self.model_type == "logistic_regression":
            self.scaler = StandardScaler()
            X_train_proc = self.scaler.fit_transform(X_imputed)
        else:
            X_train_proc = X_imputed
            
        self.model_ = self._init_estimator(scale_pos_weight=scale_pos_weight)
        self.model_.fit(X_train_proc, y_arr)
        self.is_fitted_ = True
        return self

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")
        # Ensure exact column alignment
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=np.nan)
        X_imputed = self.imputer.transform(X_aligned)
        if self.scaler is not None:
            return self.scaler.transform(X_imputed)
        return X_imputed

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return positive class probability vector.
        """
        X_proc = self._transform(X)
        proba = self.model_.predict_proba(X_proc)
        # Handle binary vs edge 1-class cases
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Binary prediction at specified threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importances(self) -> pd.Series:
        """
        Return series of feature importances sorted descending.
        """
        if not self.is_fitted_:
            return pd.Series(dtype=float)
            
        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, "coef_"):
            importances = np.abs(self.model_.coef_[0])
        else:
            return pd.Series(dtype=float)
            
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)

    def save(self, filepath: str | Path) -> None:
        """
        Save serialized model bundle to disk.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> HypoClassifier:
        """
        Load serialized model bundle from disk.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
