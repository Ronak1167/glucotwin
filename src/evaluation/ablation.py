r"""
GlucoTwin — Feature Ablation Analysis
src/evaluation/ablation.py

Systematic feature group ablation study under strict chronological train/test split.
Quantifies the marginal predictive value of:
- CGM features (foundation)
- Insulin kinetics (bolus, basal, IOB proxy)
- Meal/carbohydrate absorption proxies
- Physical activity & heart rate
- Circadian & nocturnal temporal context
- Personalized baselines (expanding stats, EWM, drift velocity)

SCIENTIFIC LANGUAGE POLICY:
All metrics describe observational model performance on retrospective held-out splits.
Feature contributions denote statistical predictive associations within this cohort,
NOT causal mechanisms or clinical treatment effects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.classifier import HypoClassifier
from src.evaluation.metrics import compute_clinical_metrics

logger = logging.getLogger(__name__)

# Feature group definitions for systematic ablation
FEATURE_GROUPS = {
    "cgm_core": [
        "glucose_current",
        "glucose_mean_120min",
        "glucose_std_120min",
        "glucose_min_120min",
        "glucose_max_120min",
        "glucose_range_120min",
        "glucose_mean_60min",
        "glucose_std_60min",
        "glucose_velocity_15min",
        "glucose_velocity_30min",
        "glucose_accel_30min",
        "glucose_dist_to_70",
        "glucose_dist_to_54",
        "is_near_hypo",
        "is_low",
        "is_very_low",
        "n_low_readings_window",
        "time_below_70_min",
        "glucose_dropping_fast",
    ],
    "insulin": [
        "bolus_sum_240min",
        "bolus_sum_120min",
        "n_bolus_240min",
        "basal_rate_current",
        "low_and_high_bolus",
    ],
    "meal": [
        "carbs_sum_180min",
        "carbs_sum_60min",
        "n_meals_180min",
    ],
    "activity": [
        "steps_sum_60min",
        "hr_mean_60min",
        "hr_max_60min",
    ],
    "circadian": [
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "time_in_day_fraction",
        "is_nocturnal_heuristic",
        "nocturnal_low",
    ],
    "personal_baseline": [
        "personal_expanding_mean",
        "glucose_deviation_from_baseline",
        "personal_expanding_std",
        "personal_ewm_baseline",
        "glucose_deviation_from_ewm",
        "personal_mean_velocity",
        "velocity_deviation_from_personal",
        "personal_nocturnal_mean",
        "glucose_nocturnal_deviation",
        "personal_mean_all_time",
        "personal_std_all_time",
    ],
}


def get_ablation_subsets(all_columns: List[str]) -> Dict[str, List[str]]:
    """
    Define standard ablation feature column subsets based on available columns.
    
    Subsets:
    1. 'full_model': all available features
    2. 'no_personal_baseline': full minus personal baseline features
    3. 'no_circadian': full minus circadian and nocturnal features
    4. 'cgm_and_insulin': only CGM core and insulin features
    5. 'cgm_only': only CGM core features
    """
    cols = list(all_columns)
    personal_cols = set(FEATURE_GROUPS["personal_baseline"])
    circadian_cols = set(FEATURE_GROUPS["circadian"])
    insulin_cols = set(FEATURE_GROUPS["insulin"])
    meal_cols = set(FEATURE_GROUPS["meal"])
    activity_cols = set(FEATURE_GROUPS["activity"])

    subsets = {
        "full_model": cols,
        "no_personal_baseline": [c for c in cols if c not in personal_cols],
        "no_circadian": [c for c in cols if c not in circadian_cols],
        "cgm_and_insulin": [c for c in cols if c not in personal_cols and c not in circadian_cols and c not in meal_cols and c not in activity_cols],
        "cgm_only": [c for c in cols if c not in personal_cols and c not in circadian_cols and c not in insulin_cols and c not in meal_cols and c not in activity_cols],
    }
    return subsets


def run_feature_ablation_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.25,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate models across defined feature ablation configurations.
    
    Operates strictly on the provided train/test split.
    Zero leakage: Models trained solely on X_train/y_train; tested on X_test/y_test.
    
    Returns:
        Dict with metrics for each configuration, baseline deltas, and configuration metadata.
    """
    available_cols = [c for c in X_train.columns if c in X_test.columns]
    subsets = get_ablation_subsets(available_cols)
    
    results: Dict[str, Any] = {}
    full_metrics: Optional[Dict[str, float]] = None

    for config_name, feature_cols in subsets.items():
        if len(feature_cols) == 0:
            logger.warning(f"Feature set for configuration '{config_name}' is empty. Skipping.")
            continue
        
        logger.info(f"Running ablation configuration: '{config_name}' ({len(feature_cols)} features)")
        
        X_tr = X_train[feature_cols].copy()
        X_te = X_test[feature_cols].copy()
        
        # Train model for this ablation configuration
        model = HypoClassifier(
            model_type="lightgbm",
            n_estimators=n_estimators,
            random_state=random_state,
        )
        model.fit(X_tr, y_train)
        
        # Evaluate on held-out test split
        y_proba = model.predict_proba(X_te)
        metrics = compute_clinical_metrics(y_test, y_proba, threshold=threshold)
        
        entry = {
            "n_features": len(feature_cols),
            "features": feature_cols,
            "metrics": metrics,
        }
        
        if config_name == "full_model":
            full_metrics = metrics
            entry["f2_delta_vs_full"] = 0.0
            entry["roc_auc_delta_vs_full"] = 0.0
        elif full_metrics is not None:
            entry["f2_delta_vs_full"] = round(metrics.get("f2", 0.0) - full_metrics.get("f2", 0.0), 4)
            entry["roc_auc_delta_vs_full"] = round(metrics.get("roc_auc", 0.0) - full_metrics.get("roc_auc", 0.0), 4)
            
        results[config_name] = entry

    return {
        "ablation_results": results,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "threshold": threshold,
        "evaluation_protocol": "Chronological 80/20 train/test split, zero leakage",
    }
