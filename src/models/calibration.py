r"""
GlucoTwin — Production Probability Calibration
src/models/calibration.py

Production-safe probability calibration for digital twin risk inference.
Provides:
- IsotonicCalibrator: Monotonic non-parametric probability calibration (recommended)
- PlattCalibrator: Parametric logistic scaling calibration (sigmoid)

LEAKAGE SAFETY:
- Calibrators are fit strictly on validation split predictions and labels.
- Test data and runtime patient data are NEVER used for fitting.
- Fitting during inference is strictly prohibited.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def _validate_probabilities(p: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """
    Validate that inputs are valid probabilities in [0, 1].
    Allows small floating-point tolerance, clipping within [-tol, 1+tol].
    Raises ValueError for non-finite values or values outside [0, 1].
    """
    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities must be finite (no NaN or Inf permitted).")
    
    if np.any(p < -tol) or np.any(p > 1.0 + tol):
        raise ValueError(
            f"Probabilities must be within [0, 1]. "
            f"Observed min: {np.min(p):.4f}, max: {np.max(p):.4f}"
        )
    return np.clip(p, 0.0, 1.0)


class IsotonicCalibrator:
    """
    Production-safe Isotonic Probability Calibrator.

    Maps uncalibrated tree-model probability predictions to empirically calibrated
    probabilities using isotonic regression fit strictly on validation data.

    Properties:
    - Input probabilities must be in [0, 1].
    - Output probabilities are guaranteed in [0, 1].
    - Strictly monotonic: non-decreasing mapping preserves ranking.
    - Deterministic and thread-safe.
    - Supports serialization via joblib (save / load).
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self._regressor = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        self.is_fitted_: bool = False
        self.metadata: Dict[str, Any] = metadata or {}

    def fit(
        self,
        raw_proba: Union[np.ndarray, pd.Series, List[float]],
        y_true: Union[np.ndarray, pd.Series, List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on validation predictions and labels.

        CRITICAL: raw_proba and y_true MUST come from a validation split,
        NEVER from the test set or future patient data.
        """
        p = np.asarray(raw_proba, dtype=float).ravel()
        y = np.asarray(y_true, dtype=int).ravel()

        if len(p) != len(y):
            raise ValueError(f"Lengths of raw_proba ({len(p)}) and y_true ({len(y)}) must match.")
        if len(p) < 10:
            raise ValueError("At least 10 samples are required to fit calibration.")

        p_valid = _validate_probabilities(p)
        self._regressor.fit(p_valid, y)
        self.is_fitted_ = True

        if metadata:
            self.metadata.update(metadata)

        return self

    def transform(
        self,
        raw_proba: Union[float, int, np.ndarray, pd.Series, List[float]],
    ) -> Union[float, np.ndarray]:
        """
        Transform raw probabilities into calibrated probabilities.

        Accepts scalar float or array-like. Returns matching type.
        Output is strictly bounded to [0.0, 1.0].
        """
        if not self.is_fitted_:
            raise RuntimeError("Calibrator has not been fitted. Call fit() or load() an artifact.")

        is_scalar = isinstance(raw_proba, (float, int, np.floating, np.integer))
        if is_scalar:
            p_arr = np.array([float(raw_proba)], dtype=float)
        else:
            p_arr = np.asarray(raw_proba, dtype=float)

        orig_shape = p_arr.shape
        p_flat = p_arr.ravel()
        p_valid = _validate_probabilities(p_flat)

        cal_flat = self._regressor.predict(p_valid)
        cal_flat = np.clip(cal_flat, 0.0, 1.0)

        if is_scalar:
            return float(cal_flat[0])
        return cal_flat.reshape(orig_shape)

    def calibrate(
        self,
        raw_proba: Union[float, int, np.ndarray, pd.Series, List[float]],
    ) -> Union[float, np.ndarray]:
        """Alias for transform() for backward compatibility with PlattCalibrator API."""
        return self.transform(raw_proba)

    def compute_calibration_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        raw_proba: Union[np.ndarray, pd.Series],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute calibration curve, Brier scores, and Expected Calibration Error (ECE).
        """
        calibrated = self.transform(raw_proba)
        y = np.asarray(y_true, dtype=float).ravel()
        raw = np.asarray(raw_proba, dtype=float).ravel()
        cal = np.asarray(calibrated, dtype=float).ravel()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_sum = 0.0
        calibration_bins = []
        n_total = len(y)

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (cal >= lo) & (cal < hi)
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            mean_pred = float(cal[mask].mean())
            mean_actual = float(y[mask].mean())
            ece_sum += (n_bin / n_total) * abs(mean_pred - mean_actual)
            calibration_bins.append({
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "n": int(n_bin),
                "mean_predicted": round(mean_pred, 3),
                "mean_actual": round(mean_actual, 3),
                "gap": round(mean_pred - mean_actual, 3),
            })

        return {
            "brier_raw": round(float(brier_score_loss(y, raw)), 4),
            "brier_calibrated": round(float(brier_score_loss(y, cal)), 4),
            "ece": round(float(ece_sum), 4),
            "calibration_bins": calibration_bins,
            "n_bins": n_bins,
            "method": "isotonic",
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize calibrator bundle to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"IsotonicCalibrator saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IsotonicCalibrator":
        """Load serialized calibrator bundle from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibrator file not found: {filepath}")
        loaded = joblib.load(filepath)
        if isinstance(loaded, cls):
            return loaded
        elif isinstance(loaded, dict) and "calibrator" in loaded:
            cal = loaded["calibrator"]
            if hasattr(cal, "metadata") and not cal.metadata:
                cal.metadata = loaded.get("metadata", {})
            return cal
        elif hasattr(loaded, "calibrate") or hasattr(loaded, "transform"):
            return loaded
        raise TypeError(f"Unrecognized calibrator object type: {type(loaded)}")


class PlattCalibrator(IsotonicCalibrator):
    """
    Parametric or Non-parametric probability calibrator.
    Maintains full backward compatibility with src/evaluation/lopo.py.
    """

    def __init__(self, method: str = "platt", metadata: Optional[Dict[str, Any]] = None):
        super().__init__(metadata=metadata)
        self.method = method.lower()
        if self.method == "platt":
            self._regressor = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        elif self.method == "isotonic":
            self._regressor = IsotonicRegression(
                increasing=True,
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def fit(
        self,
        raw_proba: Union[np.ndarray, pd.Series, List[float]],
        y_true: Union[np.ndarray, pd.Series, List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PlattCalibrator":
        p = np.asarray(raw_proba, dtype=float).ravel()
        y = np.asarray(y_true, dtype=int).ravel()

        if len(p) != len(y):
            raise ValueError(f"Lengths of raw_proba ({len(p)}) and y_true ({len(y)}) must match.")

        p_valid = _validate_probabilities(p)

        if self.method == "platt":
            self._regressor.fit(p_valid.reshape(-1, 1), y)
        else:
            self._regressor.fit(p_valid, y)

        self.is_fitted_ = True
        if metadata:
            self.metadata.update(metadata)
        return self

    def transform(
        self,
        raw_proba: Union[float, int, np.ndarray, pd.Series, List[float]],
    ) -> Union[float, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Calibrator not fitted.")

        is_scalar = isinstance(raw_proba, (float, int, np.floating, np.integer))
        p_arr = np.array([float(raw_proba)], dtype=float) if is_scalar else np.asarray(raw_proba, dtype=float)
        orig_shape = p_arr.shape
        p_flat = p_arr.ravel()
        p_valid = _validate_probabilities(p_flat)

        if self.method == "platt":
            cal_flat = self._regressor.predict_proba(p_valid.reshape(-1, 1))[:, 1]
        else:
            cal_flat = self._regressor.predict(p_valid)

        cal_flat = np.clip(cal_flat, 0.0, 1.0)
        if is_scalar:
            return float(cal_flat[0])
        return cal_flat.reshape(orig_shape)
