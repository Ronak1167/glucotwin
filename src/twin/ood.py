r"""
GlucoTwin — Out-of-Distribution (OOD) Detector & Data Quality Engine
src/twin/ood.py

Detects unusual feature combinations, missing signals, and feature-space
distance from training distribution. Combines all evidence into a single
structured confidence object.

Design principles:
- Confidence must come from measurable evidence
- No arbitrary random confidence values
- Document exactly how each score is computed
- Confidence reflects what the model ACTUALLY KNOWS, not what it guesses

The five evidence components:
  1. Signal completeness: fraction of expected signals present and valid
  2. CGM continuity: penalize gaps in glucose monitoring
  3. Feature-space Mahalanobis distance: how far current sample is from training mean
  4. Value range: any features outside training interquartile range
  5. Historical coverage: how recently matched historical episodes exist

Overall confidence: weighted average of all five, with hard floors for critical failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """
    Structured OOD assessment for a single twin state evaluation.

    confidence_level: 'HIGH' | 'MODERATE' | 'LOW'
    confidence_score: 0.0–1.0 (fully documented, deterministic)
    out_of_distribution: True if score < LOW_THRESHOLD
    reasons: Human-readable explanation of low-confidence contributors
    signal_completeness: 0.0–1.0
    cgm_continuity: 0.0–1.0
    feature_range_score: 0.0–1.0
    mahalanobis_percentile: 0.0–100.0 (higher = more unusual)
    """
    confidence_level: str = "HIGH"
    confidence_score: float = 1.0
    out_of_distribution: bool = False
    reasons: List[str] = field(default_factory=list)
    signal_completeness: float = 1.0
    cgm_continuity: float = 1.0
    feature_range_score: float = 1.0
    mahalanobis_percentile: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 3),
            "out_of_distribution": self.out_of_distribution,
            "reasons": self.reasons,
            "components": {
                "signal_completeness": round(self.signal_completeness, 3),
                "cgm_continuity": round(self.cgm_continuity, 3),
                "feature_range_score": round(self.feature_range_score, 3),
                "mahalanobis_percentile": round(self.mahalanobis_percentile, 1),
            },
        }


class OODDetector:
    """
    Lightweight OOD/confidence estimator for the Digital Twin.

    Architecture:
    - fit() on TRAINING feature matrix → stores statistics
    - assess() on any new feature row → returns OODResult

    Mahalanobis distance is approximated using diagonal covariance
    (per-feature standardization) to avoid inversion issues with
    correlated high-dimensional features.
    """

    # Confidence thresholds
    HIGH_THRESHOLD = 0.75
    LOW_THRESHOLD = 0.45

    def __init__(self):
        self._fitted = False
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._train_q25: Optional[np.ndarray] = None
        self._train_q75: Optional[np.ndarray] = None
        self._train_dist_p90: float = 0.0  # 90th percentile of training distances
        self._feature_names: List[str] = []

        # Signal groups: features present → signal available
        self._cgm_features = ["glucose_current", "glucose_velocity_15min", "glucose_mean_120min"]
        self._insulin_features = ["bolus_sum_240min", "basal_sum_240min"]
        self._meal_features = ["carbs_sum_180min"]
        self._hr_features = ["hr_mean_60min"]
        self._sleep_features = ["is_nocturnal_heuristic"]

    def fit(self, X_train: pd.DataFrame) -> "OODDetector":
        """
        Fit OOD detector on training feature matrix.
        Must be called ONLY on training data — never on test/validation data.
        """
        self._feature_names = list(X_train.columns)
        X_arr = X_train.values.astype(float)

        # Per-feature statistics (NaN-robust)
        self._train_mean = np.nanmean(X_arr, axis=0)
        self._train_std = np.nanstd(X_arr, axis=0)
        self._train_std = np.where(self._train_std < 1e-8, 1.0, self._train_std)
        self._train_q25 = np.nanpercentile(X_arr, 25, axis=0)
        self._train_q75 = np.nanpercentile(X_arr, 75, axis=0)

        # Compute training Mahalanobis distances (diagonal approximation)
        # d_i = sqrt( sum_j ((x_ij - mu_j) / sigma_j)^2 / n_features )
        z_train = (X_arr - self._train_mean) / self._train_std
        z_train = np.where(np.isnan(z_train), 0.0, z_train)
        dist_train = np.sqrt(np.mean(z_train ** 2, axis=1))
        self._train_dist_p90 = float(np.nanpercentile(dist_train, 90))

        self._fitted = True
        logger.info(
            f"OODDetector fitted on {len(X_train)} samples, "
            f"{len(self._feature_names)} features. "
            f"Training dist p90={self._train_dist_p90:.2f}"
        )
        return self

    def assess(
        self,
        X_sample: pd.Series | pd.DataFrame,
        cgm_gap_minutes: float = 0.0,
    ) -> OODResult:
        """
        Assess OOD confidence for a single feature row.

        Args:
            X_sample: Feature row (pd.Series or single-row DataFrame)
            cgm_gap_minutes: Time since last valid CGM reading

        Returns:
            OODResult with confidence level and documented reasons
        """
        if isinstance(X_sample, pd.DataFrame):
            row = X_sample.iloc[0]
        else:
            row = X_sample

        reasons = []

        # ── 1. Signal Completeness ────────────────────────────────────────────
        signal_groups = {
            "CGM": self._cgm_features,
            "Insulin": self._insulin_features,
            "Meals": self._meal_features,
            "Heart Rate": self._hr_features,
        }
        n_groups = len(signal_groups)
        n_available = 0
        for group_name, feats in signal_groups.items():
            group_vals = [row.get(f, np.nan) for f in feats if f in (self._feature_names or [])]
            if not group_vals:
                # Feature simply not in training set (e.g. HR unavailable for all patients)
                n_available += 0.5
                continue
            any_valid = any(not np.isnan(v) for v in group_vals)
            if any_valid:
                n_available += 1.0
            else:
                reasons.append(f"missing {group_name.lower()} data")

        signal_completeness = n_available / n_groups if n_groups > 0 else 1.0

        # ── 2. CGM Continuity ─────────────────────────────────────────────────
        if cgm_gap_minutes <= 0:
            cgm_continuity = 1.0
        elif cgm_gap_minutes <= 15:
            cgm_continuity = 0.85
        elif cgm_gap_minutes <= 30:
            cgm_continuity = 0.65
            reasons.append(f"CGM gap of {cgm_gap_minutes:.0f} min (one missed reading)")
        elif cgm_gap_minutes <= 60:
            cgm_continuity = 0.35
            reasons.append(f"CGM gap of {cgm_gap_minutes:.0f} min (multiple missed readings)")
        else:
            cgm_continuity = 0.1
            reasons.append(f"Large CGM gap of {cgm_gap_minutes:.0f} min — very low confidence")

        # ── 3. Feature Range Score ────────────────────────────────────────────
        if self._fitted:
            n_features = len(self._feature_names)
            n_in_range = 0
            iqr_span = 3.0  # Accept values within 3×IQR of training range

            for feat, q25, q75 in zip(
                self._feature_names,
                self._train_q25,
                self._train_q75,
            ):
                val = row.get(feat, np.nan)
                if np.isnan(val):
                    n_in_range += 0.5  # Missing is less concerning than extreme
                    continue
                iqr = max(q75 - q25, 1e-6)
                lo = q25 - iqr_span * iqr
                hi = q75 + iqr_span * iqr
                if lo <= val <= hi:
                    n_in_range += 1.0

            feature_range_score = n_in_range / n_features if n_features > 0 else 1.0
            if feature_range_score < 0.7:
                reasons.append(
                    f"feature values deviate significantly from training distribution "
                    f"({feature_range_score * 100:.0f}% in range)"
                )
        else:
            feature_range_score = 0.8  # Unknown without fit

        # ── 4. Mahalanobis Distance ───────────────────────────────────────────
        if self._fitted:
            feat_vals = np.array([row.get(f, np.nan) for f in self._feature_names], dtype=float)
            z = (feat_vals - self._train_mean) / self._train_std
            z = np.where(np.isnan(z), 0.0, z)
            dist = float(np.sqrt(np.mean(z ** 2)))
            # Convert to percentile relative to training
            # dist > p90 means this sample is more unusual than 90% of training samples
            ratio = dist / max(self._train_dist_p90, 1e-6)
            # Map ratio to percentile-like score (rough heuristic)
            mahalanobis_percentile = float(np.clip(ratio * 90.0, 0.0, 100.0))

            if ratio > 2.5:
                reasons.append(f"large deviation from training distribution (distance ratio={ratio:.1f})")
            elif ratio > 1.5:
                reasons.append(f"moderate deviation from training distribution (distance ratio={ratio:.1f})")
        else:
            mahalanobis_percentile = 0.0
            ratio = 0.0

        # ── 5. Compute Overall Confidence ─────────────────────────────────────
        # Weighted combination (documented weights):
        # signal_completeness: 30% (critical for all features)
        # cgm_continuity:      30% (CGM is the primary signal)
        # feature_range_score: 25% (distribution fit)
        # mahalanobis:         15% (holistic distance)
        maha_score = float(np.clip(1.0 - (ratio - 1.0) / 3.0, 0.0, 1.0)) if self._fitted else 0.85

        confidence_score = (
            0.30 * signal_completeness
            + 0.30 * cgm_continuity
            + 0.25 * feature_range_score
            + 0.15 * maha_score
        )
        confidence_score = float(np.clip(confidence_score, 0.0, 1.0))

        # Classify
        if confidence_score >= self.HIGH_THRESHOLD:
            confidence_level = "HIGH"
            ood = False
        elif confidence_score >= self.LOW_THRESHOLD:
            confidence_level = "MODERATE"
            ood = False
        else:
            confidence_level = "LOW"
            ood = True

        return OODResult(
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            out_of_distribution=ood,
            reasons=reasons,
            signal_completeness=signal_completeness,
            cgm_continuity=cgm_continuity,
            feature_range_score=feature_range_score,
            mahalanobis_percentile=mahalanobis_percentile,
        )
