r"""
GlucoTwin — Evaluation Metrics
src/evaluation/metrics.py

Clinically grounded evaluation metrics for hypoglycemia prediction.
Includes sensitivity (recall), specificity, precision, F1, F2 (recall-weighted),
ROC-AUC, PR-AUC, and nocturnal subgroup analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_clinical_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    threshold: float = 0.5,
    sample_interval_min: float = 5.0,
) -> Dict[str, float]:
    """
    Compute comprehensive clinical and statistical classification metrics.
    
    Args:
        y_true: Ground truth binary labels (1 = hypo, 0 = non-hypo)
        y_pred_proba: Predicted probabilities for class 1
        threshold: Decision threshold for positive prediction
        sample_interval_min: Sampling interval in minutes (5 min for HUPA-UCM)
        
    Returns:
        Dict of metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics: Dict[str, float] = {}
    
    n_samples = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_samples - n_pos
    pos_rate = float(n_pos / n_samples) if n_samples > 0 else 0.0
    
    metrics["n_samples"] = float(n_samples)
    metrics["n_positive"] = float(n_pos)
    metrics["positive_prevalence"] = pos_rate
    metrics["threshold_used"] = float(threshold)
    
    # Check if both classes are present
    if n_pos == 0 or n_neg == 0:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = pos_rate
        metrics["sensitivity"] = 0.0
        metrics["specificity"] = 1.0 if n_pos == 0 else 0.0
        metrics["precision"] = 0.0
        metrics["f1"] = 0.0
        metrics["f2"] = 0.0
        metrics["false_alarm_rate_per_day"] = 0.0
        return metrics
        
    # Standard statistical metrics
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics["roc_auc"] = 0.5
        
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        metrics["pr_auc"] = pos_rate
        
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Clinical Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (Catching hypos)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Non-hypo correctness
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # True alarm fraction
    
    metrics["sensitivity"] = float(sensitivity)
    metrics["recall"] = float(sensitivity)
    metrics["specificity"] = float(specificity)
    metrics["precision"] = float(precision)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["f2"] = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
    
    # Daily false alarm burden (crucial to avoid patient alarm fatigue)
    # Samples per day = (24 * 60) / sample_interval_min = 288 for 5-min intervals
    samples_per_day = 1440.0 / sample_interval_min
    days_observed = n_samples / samples_per_day if samples_per_day > 0 else 1.0
    metrics["false_alarm_rate_per_day"] = float(fp / days_observed) if days_observed > 0 else 0.0
    
    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["tn"] = float(tn)
    metrics["fn"] = float(fn)
    
    return metrics


def compute_subgroup_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    subgroup_mask: np.ndarray | pd.Series,
    subgroup_name: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics restricted to a specific subgroup (e.g. Nocturnal).
    """
    mask = np.asarray(subgroup_mask).astype(bool)
    if not mask.any():
        logger.warning(f"Subgroup {subgroup_name} has 0 samples.")
        return {}
        
    sub_true = np.asarray(y_true)[mask]
    sub_proba = np.asarray(y_pred_proba)[mask]
    
    sub_metrics = compute_clinical_metrics(sub_true, sub_proba, threshold=threshold)
    return {f"{subgroup_name}_{k}": v for k, v in sub_metrics.items()}


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    target_metric: str = "f2",
    min_sensitivity: float = 0.70,
    max_false_alarms_day: float = 5.0,
    sample_interval_min: float = 5.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Search thresholds from 0.05 to 0.95 to find clinically optimal operating point.
    Default targets F2 score (penalizing false negatives more heavily than false positives)
    while enforcing minimum sensitivity constraint.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}
    
    for thresh in np.linspace(0.05, 0.90, 86):
        m = compute_clinical_metrics(
            y_true, y_pred_proba, threshold=thresh, sample_interval_min=sample_interval_min
        )
        score = m.get(target_metric, 0.0)
        
        # Check clinical viability
        if m.get("sensitivity", 0.0) >= min_sensitivity:
            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_metrics = m
                
    if not best_metrics:
        # Fallback to unconstrained maximum F1/F2
        for thresh in np.linspace(0.05, 0.90, 86):
            m = compute_clinical_metrics(
                y_true, y_pred_proba, threshold=thresh, sample_interval_min=sample_interval_min
            )
            score = m.get(target_metric, 0.0)
            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_metrics = m
                
    return float(best_thresh), best_metrics
