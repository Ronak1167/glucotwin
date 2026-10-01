r"""
GlucoTwin — Explainability Engine
src/explainability/explainer.py

Provides SHAP-based feature attributions and clinician/patient-friendly natural language
narratives explaining why a hypoglycemia warning was triggered.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class TwinExplainer:
    """
    SHAP-based explainer with clinical narrative synthesis.
    """
    def __init__(self, classifier_bundle: Any):
        self.clf = classifier_bundle
        self.explainer: Optional[Any] = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            # TreeExplainer for Tree models
            if self.clf.model_type in ["lightgbm", "xgboost", "random_forest"]:
                self.explainer = shap.TreeExplainer(self.clf.model_)
            else:
                self.explainer = None
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP TreeExplainer: {e}")
            self.explainer = None

    def explain_sample(
        self,
        X_sample: pd.DataFrame | pd.Series,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain a single observation at time T.
        
        Returns:
            Dict containing top positive contributors (risk drivers) and negative contributors (protective factors),
            along with a generated clinical explanation narrative.
        """
        if isinstance(X_sample, pd.Series):
            df_sample = X_sample.to_frame().T
        else:
            df_sample = X_sample.copy()

        # Transform features using classifier pipeline
        X_proc = self.clf._transform(df_sample)
        feature_names = self.clf.feature_names_

        if self.explainer is not None:
            try:
                # Suppress the SHAP LightGBM binary output-shape deprecation warning
                # (we already handle both old and new output shapes below)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="LightGBM binary classifier with TreeExplainer",
                        category=UserWarning,
                    )
                    shap_vals = self.explainer.shap_values(X_proc)
                # Handle all SHAP binary classifier output shapes:
                # Old SHAP (<0.41):  list of 2 ndarrays, each shape (n_samples, n_features)
                # New SHAP (>=0.41): list of 2 ndarrays (same), or single 3D ndarray
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    # Both old and new LightGBM binary output — take positive class [1]
                    raw = shap_vals[1]
                    vals = raw[0] if raw.ndim == 2 else raw.flatten()
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # 3D: (n_samples, n_features, n_classes) — take positive class
                    vals = shap_vals[0, :, 1]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                    # 2D single-output model
                    vals = shap_vals[0]
                else:
                    vals = np.array(shap_vals).flatten()
            except Exception as e:
                logger.warning(f"SHAP calculation error: {e}. Falling back to feature importances.")
                vals = None
        else:
            vals = None

        # Fallback to feature values and global importances if SHAP fails
        if vals is None:
            imp = self.clf.get_feature_importances()
            top_features = []
            for col, importance in imp.head(top_k).items():
                val = float(df_sample[col].iloc[0]) if col in df_sample.columns else 0.0
                top_features.append({
                    "feature": col,
                    "importance": round(float(importance), 4),
                    "value": round(val, 2),
                    "impact": "risk_driver" if val < 80 else "context",
                })
            return {
                "top_risk_factors": top_features,
                "clinical_narrative": self._generate_fallback_narrative(df_sample),
            }

        # Structure SHAP factors
        contributions = []
        for name, shap_val, raw_val in zip(feature_names, vals, df_sample.iloc[0].values):
            contributions.append({
                "feature": name,
                "shap_value": float(shap_val),
                "value": float(raw_val) if pd.notna(raw_val) else None,
                "direction": "increases_hypo_risk" if shap_val > 0 else "reduces_hypo_risk",
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_factors = contributions[:top_k]

        narrative = self._generate_clinical_narrative(df_sample, top_factors)

        return {
            "top_risk_factors": top_factors,
            "clinical_narrative": narrative,
        }

    def _generate_clinical_narrative(
        self,
        df_sample: pd.DataFrame,
        top_factors: List[Dict[str, Any]],
    ) -> str:
        """
        Synthesize human-readable clinical explanation based on patient state and SHAP drivers.
        """
        row = df_sample.iloc[0]
        glucose = row.get("glucose_current", None)
        vel = row.get("glucose_velocity_15min", None)
        bolus = row.get("bolus_sum_240min", 0.0)
        is_nocturnal = row.get("is_nocturnal_heuristic", 0)

        narrative_parts = []

        # Current glucose context
        if glucose is not None and not np.isnan(glucose):
            if glucose < 70:
                narrative_parts.append(f"Current glucose is critically low at {glucose:.0f} mg/dL (Level 1 hypoglycemia).")
            elif glucose < 90:
                narrative_parts.append(f"Current glucose is in the near-hypoglycemia warning zone at {glucose:.0f} mg/dL.")
            else:
                narrative_parts.append(f"Current glucose is {glucose:.0f} mg/dL.")

        # Rate of change
        if vel is not None and not np.isnan(vel):
            if vel < -1.5:
                narrative_parts.append(f"Glucose is dropping rapidly at {abs(vel):.2f} mg/dL per minute.")
            elif vel < -0.5:
                narrative_parts.append(f"Glucose is drifting downwards at {abs(vel):.2f} mg/dL per minute.")

        # Insulin impact
        if bolus and bolus > 0.5:
            narrative_parts.append(f"Recent active bolus insulin of {bolus:.1f} Units represents significant ongoing downward pressure.")

        # Nocturnal context
        if is_nocturnal == 1:
            narrative_parts.append("Nocturnal period (sleep phase) reduces patient symptom awareness and counter-regulatory response.")

        # Key drivers from SHAP
        driver_names = [f["feature"].replace("_", " ") for f in top_factors if f["shap_value"] > 0][:3]
        if driver_names:
            narrative_parts.append(f"Primary risk drivers identified by twin: {', '.join(driver_names)}.")

        return " ".join(narrative_parts)

    def _generate_fallback_narrative(self, df_sample: pd.DataFrame) -> str:
        row = df_sample.iloc[0]
        g = row.get("glucose_current", 100)
        vel = row.get("glucose_velocity_15min", 0)
        return (
            f"Patient state: glucose {g:.1f} mg/dL, 15-min velocity {vel:.2f} mg/dL/min. "
            f"Hypoglycemia risk computed from metabolic trajectory and temporal trends."
        )
