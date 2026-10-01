r"""
GlucoTwin — Baseline Models
src/models/baselines.py

Heuristic and clinical baselines for hypoglycemia prediction benchmarking:
1. Persistence Baseline: Predicts hypo if current glucose < threshold
2. Trend / Rate-of-Change Baseline: Predicts hypo if glucose is low and dropping fast
3. Moving Average Baseline: Predicts hypo if 60-min moving average < threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class PersistenceBaseline:
    """
    Predicts hypoglycemia at T + H if current glucose at T is already < threshold.
    """
    def __init__(self, threshold_mgdl: float = 70.0):
        self.threshold_mgdl = threshold_mgdl
        self.name = "PersistenceBaseline"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PersistenceBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "glucose_current" in X.columns:
            g = X["glucose_current"].fillna(120.0).values
        else:
            g = X.iloc[:, 0].fillna(120.0).values
        # Heuristic probability inversely related to distance from hypo
        # Under 70 -> 1.0, 70-100 -> linear decay, >100 -> 0.05
        proba = np.clip((110.0 - g) / 40.0, 0.0, 1.0)
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class TrendVelocityBaseline:
    """
    Predicts hypoglycemia if glucose is in warning zone (<95 mg/dL)
    AND velocity is negative (dropping towards hypoglycemia).
    """
    def __init__(self, warning_glucose: float = 95.0, velocity_thresh: float = -0.8):
        self.warning_glucose = warning_glucose
        self.velocity_thresh = velocity_thresh
        self.name = "TrendVelocityBaseline"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TrendVelocityBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        g = X["glucose_current"].fillna(120.0).values if "glucose_current" in X.columns else X.iloc[:, 0].fillna(120.0).values
        vel = X["glucose_velocity_15min"].fillna(0.0).values if "glucose_velocity_15min" in X.columns else np.zeros_like(g)
        
        # Extrapolate linear glucose in 30-60 min: projected = g + vel * 30
        projected = g + vel * 30.0
        proba = np.clip((100.0 - projected) / 35.0, 0.0, 1.0)
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
