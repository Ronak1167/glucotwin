r"""
GlucoTwin — Historical Episode Similarity Matcher
src/similarity/matcher.py

Finds top-K historical episodes from a patient's own history matching the current
metabolic and physiological pattern (glucose level, velocity, IOB, circadian time).
Provides grounded real-world precedents for digital twin decisions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalEpisodeMatcher:
    """
    Retrieves most similar historical episodes using multidimensional feature similarity.
    """
    def __init__(
        self,
        top_k: int = 3,
        min_temporal_separation_hours: float = 6.0,
    ):
        self.top_k = top_k
        self.min_sep_hours = min_temporal_separation_hours
        self.history_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None

    def index_history(self, preprocessed_df: pd.DataFrame, feature_matrix: pd.DataFrame) -> None:
        """
        Index historical records for fast retrieval.
        """
        self.history_df = preprocessed_df.copy()
        self.feature_matrix = feature_matrix.copy()

    def find_similar_episodes(
        self,
        current_features: pd.Series | Dict[str, Any],
        current_time: Optional[pd.Timestamp] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find top-K most similar historical time points before current_time.
        """
        if self.feature_matrix is None or self.history_df is None:
            return []

        k = top_k or self.top_k
        
        # Key comparison dimensions
        key_cols = [
            ("glucose_current", 40.0),             # Norm scale ~ 40 mg/dL
            ("glucose_velocity_15min", 1.0),       # Norm scale ~ 1.0 mg/dL/min
            ("bolus_sum_240min", 2.0),             # Norm scale ~ 2 Units
            ("hour_sin", 1.0),                     # Circadian phase
            ("hour_cos", 1.0),
            ("is_nocturnal_heuristic", 1.0),
        ]

        # Valid candidate columns present in history
        valid_cols = [(col, weight) for col, weight in key_cols if col in self.feature_matrix.columns]
        if not valid_cols:
            return []

        # Historical mask: strictly before current_time (prevent leakage)
        if current_time is not None and "timestamp" in self.history_df.columns:
            ts = pd.to_datetime(self.history_df["timestamp"])
            cand_mask = ts < (current_time - pd.Timedelta(hours=self.min_sep_hours))
        else:
            cand_mask = pd.Series(True, index=self.feature_matrix.index)

        # Require candidates to have valid glucose
        cand_indices = self.feature_matrix[cand_mask].index
        if len(cand_indices) == 0:
            return []

        # Compute normalized squared distance
        dist_sq = np.zeros(len(cand_indices), dtype=float)
        for col, norm_scale in valid_cols:
            curr_val = current_features.get(col, np.nan)
            if np.isnan(curr_val):
                continue
            hist_vals = self.feature_matrix.loc[cand_indices, col].fillna(curr_val).values
            diff = (hist_vals - curr_val) / norm_scale
            dist_sq += (diff ** 2)

        distances = np.sqrt(dist_sq)
        
        # Sort candidates
        sorted_pos = np.argsort(distances)
        
        # Temporal deduplication: ensure returned episodes are at least min_sep_hours apart
        selected_episodes = []
        selected_times = []
        
        for pos in sorted_pos:
            idx = cand_indices[pos]
            dist = float(distances[pos])
            row_hist = self.history_df.loc[idx]
            
            t_hist = pd.to_datetime(row_hist.get("timestamp", None))
            
            # Check temporal separation from already selected episodes
            too_close = False
            if t_hist is not None:
                for t_prev in selected_times:
                    if abs((t_hist - t_prev).total_seconds()) / 3600.0 < self.min_sep_hours:
                        too_close = True
                        break
            if too_close:
                continue

            if t_hist is not None:
                selected_times.append(t_hist)

            # Look forward 2 hours from this historical point to see outcome
            outcome_nadir = float(row_hist.get("glucose_mgdl", 100))
            hypo_occurred = bool(outcome_nadir < 70)
            
            if "timestamp" in self.history_df.columns:
                t_fwd_end = t_hist + pd.Timedelta(minutes=120)
                mask_fwd = (pd.to_datetime(self.history_df["timestamp"]) > t_hist) & (pd.to_datetime(self.history_df["timestamp"]) <= t_fwd_end)
                if mask_fwd.any():
                    fwd_gluc = self.history_df.loc[mask_fwd, "glucose_mgdl"].dropna()
                    if len(fwd_gluc) > 0:
                        outcome_nadir = float(fwd_gluc.min())
                        hypo_occurred = bool(outcome_nadir < 70.0)

            # Similarity score 0-100%
            similarity_pct = round(float(np.exp(-dist / 3.0) * 100.0), 1)

            selected_episodes.append({
                "episode_index": int(idx),
                "timestamp": str(t_hist) if t_hist is not None else None,
                "similarity_score_pct": similarity_pct,
                "historical_glucose": round(float(row_hist.get("glucose_mgdl", 0)), 1),
                "subsequent_nadir_mgdl": round(outcome_nadir, 1),
                "hypo_occurred": hypo_occurred,
                "summary": (
                    f"Historical episode from {t_hist.strftime('%b %d, %H:%M') if t_hist else 'past'}: "
                    f"Glucose started at {row_hist.get('glucose_mgdl', 0):.0f} mg/dL, reached nadir of {outcome_nadir:.0f} mg/dL "
                    f"({'developed hypoglycemia' if hypo_occurred else 'stayed in safe range'})."
                )
            })

            if len(selected_episodes) >= k:
                break

        return selected_episodes
