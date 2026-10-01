r"""
GlucoTwin — Counterfactual What-If Simulation Engine (v2)
src/simulation/simulator.py

Simulates prospective physiological trajectories under altered input conditions.

SCIENTIFIC LANGUAGE POLICY:
This module produces COMPUTATIONAL SIMULATIONS under altered input parameters.
- "simulated_threshold_crossing_prevented" does NOT mean hypoglycemia was clinically prevented
- "counterfactual trajectory" describes a modeled scenario, not a real-world prediction
- All outputs should be interpreted as model-derived scenario exploration

Simulator model:
  Baseline: damped velocity + IOB exponential clearance
  Carb effect: gamma-shaped rise with absorption half-life (fast: 25min, complex: 55min)
  Insulin effect: rapid-acting peak at 65min (biexponential)
  Exercise: linear glucose decrease at modeled intensity rates

Fixed parameters (population averages for Type 1 Diabetes adults):
  ISF = 45 mg/dL per Unit (Insulin Sensitivity Factor)
  CR = 12 g/Unit (Carb Ratio)
  CIR = 3.75 mg/dL per gram (Carb Impact Rate = ISF/CR)
  Velocity decay constant k = 0.015/min (velocity half-life ~46 min)

Model limitations:
  - Parameters are FIXED population averages, not personalized
  - Exercise model is a simplified linear approximation
  - IOB clearance does not account for delayed gastric emptying
  - Simulations are only valid within the demonstrated physiological range
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MetabolicSimulator:
    """
    Physiological trajectory and what-if computational simulator.

    Uses fixed population-average parameters. Not a personalized model.
    All outputs are model-derived computational scenarios.
    """

    def __init__(
        self,
        nominal_isf: float = 45.0,
        nominal_cr: float = 12.0,
        nominal_cir_factor: float = 3.75,
        velocity_decay_constant: float = 0.015,
    ):
        self.isf = nominal_isf
        self.cr = nominal_cr
        self.carb_factor = nominal_cir_factor
        self.decay_k = velocity_decay_constant

    def simulate_trajectory(
        self,
        current_glucose: float,
        current_velocity: float,
        active_iob: float = 0.0,
        horizon_minutes: int = 120,
        step_minutes: int = 5,
        carb_intervention_g: float = 0.0,
        carb_type: str = "fast",
        bolus_intervention_u: float = 0.0,
        exercise_intensity: str = "none",
        threshold_mgdl: float = 70.0,
    ) -> Dict[str, Any]:
        """
        Simulate prospective glucose trajectory from time T=0 to horizon_minutes.

        Returns both baseline trajectory and counterfactual trajectory under
        specified intervention conditions.

        All fields use simulation-accurate language:
        - "simulated_nadir" not "actual minimum"
        - "simulated_threshold_crossing_prevented" not "hypoglycemia prevented"
        - "counterfactual_scenario" not "treatment"
        """
        n_steps = horizon_minutes // step_minutes
        times = np.array([i * step_minutes for i in range(n_steps + 1)], dtype=float)

        # ── 1. Baseline Natural Trajectory ──────────────────────────────────
        # Velocity naturally dampens toward 0 over time (biological mean-reversion)
        vel_trajectory = current_velocity * np.exp(-self.decay_k * times)
        baseline_change = np.cumsum(vel_trajectory * step_minutes)
        baseline_change[0] = 0.0

        # Existing active IOB clears exponentially over ~60min per unit
        iob_drop = active_iob * self.isf * (1.0 - np.exp(-times / 60.0))

        glucose_baseline = current_glucose + baseline_change - (iob_drop * 0.4)
        glucose_baseline = np.clip(glucose_baseline, 35.0, 450.0)

        # ── 2. Counterfactual Intervention ───────────────────────────────────
        carb_effect = np.zeros_like(times)
        if carb_intervention_g > 0:
            total_rise = carb_intervention_g * self.carb_factor
            if carb_type == "fast":
                # Fast absorption: gamma-like peak ~25 min
                carb_effect = total_rise * (1.0 - np.exp(-((times / 25.0) ** 1.8)))
            else:
                # Complex carb: slower sustained release, peak ~55 min
                carb_effect = total_rise * (1.0 - np.exp(-((times / 55.0) ** 1.5)))

        insulin_effect = np.zeros_like(times)
        if bolus_intervention_u > 0:
            total_drop = bolus_intervention_u * self.isf
            # Rapid-acting peak at ~65 min
            insulin_effect = total_drop * (1.0 - np.exp(-((times / 65.0) ** 2.0)))

        exercise_effect = np.zeros_like(times)
        if exercise_intensity == "light":
            exercise_effect = 22.0 * np.minimum(times / 40.0, 1.0)
        elif exercise_intensity == "moderate":
            exercise_effect = 45.0 * np.minimum(times / 40.0, 1.0)

        # ── 3. Counterfactual Trajectory ──────────────────────────────────────
        glucose_counterfactual = glucose_baseline + carb_effect - insulin_effect - exercise_effect
        glucose_counterfactual = np.clip(glucose_counterfactual, 35.0, 450.0)

        # ── 4. Threshold Crossing Detection ────────────────────────────────────
        min_baseline = float(np.min(glucose_baseline))
        min_counterfactual = float(np.min(glucose_counterfactual))

        # Baseline threshold crossing
        baseline_crosses = glucose_baseline < threshold_mgdl
        baseline_crossing_time = None
        if baseline_crosses.any():
            first_cross_idx = int(np.argmax(baseline_crosses))
            baseline_crossing_time = float(times[first_cross_idx])

        # Counterfactual threshold crossing
        cf_crosses = glucose_counterfactual < threshold_mgdl
        cf_crossing_time = None
        if cf_crosses.any():
            first_cf_idx = int(np.argmax(cf_crosses))
            cf_crossing_time = float(times[first_cf_idx])

        # Language-correct simulation flags (Python bool for JSON-safe serialization)
        baseline_any = bool(baseline_crosses.any())
        cf_any = bool(cf_crosses.any())
        simulated_threshold_crossing_prevented = bool(baseline_any and not cf_any)
        simulated_threshold_crossing_induced = bool(not baseline_any and cf_any)

        # ── 5. Trajectory Points ──────────────────────────────────────────────
        trajectory_points = []
        delta_trajectory = float(glucose_counterfactual[-1]) - float(glucose_baseline[-1])
        for t, g_base, g_cf in zip(times, glucose_baseline, glucose_counterfactual):
            trajectory_points.append({
                "minute": int(t),
                "glucose_baseline": round(float(g_base), 1),
                "glucose_simulated": round(float(g_cf), 1),
                "delta": round(float(g_cf - g_base), 1),
                "is_below_threshold_baseline": bool(g_base < threshold_mgdl),
                "is_below_threshold_simulated": bool(g_cf < threshold_mgdl),
            })

        return {
            "horizon_minutes": horizon_minutes,
            "threshold_mgdl": threshold_mgdl,
            # Baseline (no intervention)
            "baseline": {
                "simulated_nadir_mgdl": round(min_baseline, 1),
                "simulated_threshold_crossing": baseline_any,
                "simulated_crossing_time_min": baseline_crossing_time,
            },
            # Counterfactual (with intervention)
            "counterfactual": {
                "simulated_nadir_mgdl": round(min_counterfactual, 1),
                "simulated_threshold_crossing": cf_any,
                "simulated_crossing_time_min": cf_crossing_time,
                "delta_nadir_mgdl": round(min_counterfactual - min_baseline, 1),
            },
            # Scenario outcome flags (simulation-language correct)
            "simulated_threshold_crossing_prevented": simulated_threshold_crossing_prevented,
            "simulated_threshold_crossing_induced": simulated_threshold_crossing_induced,
            # Backward-compatible legacy fields
            "min_glucose_baseline": round(min_baseline, 1),
            "min_glucose_simulated": round(min_counterfactual, 1),
            "hypo_prevented": simulated_threshold_crossing_prevented,
            "hypo_induced": simulated_threshold_crossing_induced,
            # Intervention description
            "intervention": {
                "carbs_g": carb_intervention_g,
                "carb_type": carb_type,
                "bolus_u": bolus_intervention_u,
                "exercise": exercise_intensity,
                "scenario_note": (
                    "Computational simulation under altered inputs. "
                    "This does NOT constitute a clinical prediction or recommendation."
                ),
            },
            "trajectory": trajectory_points,
        }
