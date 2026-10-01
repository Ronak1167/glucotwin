r"""
GlucoTwin — Alert State Machine (Phase 9.8)
src/twin/alert.py

Implements a stateful, hysteresis-aware alert state machine for the Digital Twin.

MOTIVATION:
  A purely threshold-based alert system fires for every prediction-window crossing
  of the decision boundary, producing many independent "alert episodes" — some as
  short as one 5-minute CGM reading.  In the HUPA-UCM retrospective cohort, this
  produces ~5.26 alert episodes/patient/day even though sensitivity and specificity
  are strong.

  Root cause (investigated in Checkpoint 9.3):
  - Individual 5-minute CGM intervals are labeled independently; the model probability
    oscillates around the threshold during declining glucose trajectories.
  - Without persistence or cooldown, each threshold crossing generates a new episode.
  - This is not a calibration artifact — the raw and calibrated policies produce
    identical episode counts because the threshold equivalence preserves the boundary.

STATEFUL DESIGN:
  The state machine suppresses spurious re-alerts using:
  1. Entry persistence: N consecutive intervals must be above threshold to open an episode.
  2. Exit hysteresis: probability must fall below an exit_threshold (< entry) to close.
  3. Cooldown: minimum minutes between consecutive alert episodes (prevent oscillation).

THRESHOLDS:
  - entry_threshold = 0.44 (raw) or 0.13 (calibrated) — validation-derived
  - exit_threshold  = 0.30 (raw) or 0.08 (calibrated) — engineering default, documented
  - persistence_intervals = 1 (single interval is enough to open; avoids under-detection)
  - cooldown_minutes = 15 — one CGM interval

SCIENTIFIC LANGUAGE:
  "Alert episode" = a period during which the model sustained above-threshold predicted
  probability.  It does NOT guarantee hypoglycemia occurred.  It is a model output.

LEAKAGE GUARANTEE:
  The state machine operates purely on past and current model outputs.
  It never reads future data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AlertState(str, Enum):
    NORMAL   = "NORMAL"    # Below entry threshold
    WATCH    = "WATCH"     # Approaching — within 20% of threshold
    WARNING  = "WARNING"   # Above entry threshold, episode active
    CRITICAL = "CRITICAL"  # High probability, episode active


@dataclass
class AlertEpisode:
    """Records a single alert episode (period of sustained above-threshold probability)."""
    episode_id: int
    start_timestamp: str
    end_timestamp: Optional[str]
    peak_probability: float
    peak_alert_state: str
    n_intervals: int                  # Number of CGM intervals in episode
    resolved: bool = False
    hypo_confirmed: Optional[bool] = None  # Was actual hypo observed during/after episode?
    lead_time_to_hypo_min: Optional[float] = None


class AlertStateMachine:
    """
    Stateful, hysteresis-aware alert episode tracker.

    States: NORMAL → WATCH → WARNING/CRITICAL (→ NORMAL after cooldown)

    Thresholds:
      entry_raw        = 0.44 (validation-derived raw threshold)
      entry_calibrated = 0.13 (validation-derived calibrated equivalent)
      exit_raw         = 0.30 (engineering default — below entry to suppress oscillation)
      exit_calibrated  = 0.08 (engineering default)
      persistence      = 1 interval (one reading above threshold opens episode)
      cooldown_minutes = 15 (minimum time between episodes)

    IMPORTANT:
      All threshold values are documented with their derivation basis.
      Engineering defaults are labelled as such and are NOT called "clinically validated".
    """

    # Threshold documentation
    THRESHOLD_BASIS = {
        "entry_raw": "validation-derived (F2-maximising threshold, Checkpoint 8)",
        "entry_calibrated": "validation-derived (F2-maximising on calibrated probability, Checkpoint 9.2)",
        "exit_raw": "engineering default (0.68 × entry; prevents oscillation near boundary)",
        "exit_calibrated": "engineering default (0.62 × entry; prevents oscillation near boundary)",
        "persistence": "1 interval — single reading above threshold is sufficient",
        "cooldown": "15 minutes (one CGM interval) — minimum episode separation",
    }

    def __init__(
        self,
        entry_threshold: float = 0.44,
        exit_threshold: float = 0.30,
        persistence_intervals: int = 1,
        cooldown_minutes: float = 15.0,
        use_calibrated: bool = False,
    ):
        """
        Args:
            entry_threshold: Probability above which an alert episode opens.
                             Use 0.44 for raw, 0.13 for calibrated.
            exit_threshold: Probability below which a resolved episode is officially closed.
                            Use 0.30 for raw, 0.08 for calibrated.
            persistence_intervals: Consecutive intervals above entry before opening.
            cooldown_minutes: Minimum gap between episodes.
            use_calibrated: Whether these thresholds apply to calibrated probabilities.
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.persistence_intervals = persistence_intervals
        self.cooldown_minutes = cooldown_minutes
        self.use_calibrated = use_calibrated

        # State
        self._current_state: AlertState = AlertState.NORMAL
        self._consecutive_above: int = 0
        self._episode_active: bool = False
        self._current_episode_id: int = 0
        self._last_episode_end_ts: Optional[pd.Timestamp] = None
        self._current_episode: Optional[AlertEpisode] = None

        self._episodes: List[AlertEpisode] = []
        self._n_total_observations: int = 0

    def update(
        self,
        probability: float,
        timestamp: str | pd.Timestamp,
        risk_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process a single observation and update alert state.

        Args:
            probability: Raw or calibrated prediction probability (0..1).
            timestamp: Current observation timestamp.
            risk_score: Optional composite risk score (0..100).

        Returns:
            Dict describing current alert state and episode information.

        LEAKAGE GUARANTEE: Uses only current and past information.
        """
        self._n_total_observations += 1
        t = pd.to_datetime(timestamp)

        # ── Determine watch zone ──────────────────────────────────────────────
        watch_threshold = self.entry_threshold * 0.75

        # ── Check cooldown ────────────────────────────────────────────────────
        in_cooldown = False
        if self._last_episode_end_ts is not None:
            elapsed_min = (t - self._last_episode_end_ts).total_seconds() / 60.0
            in_cooldown = (elapsed_min < self.cooldown_minutes)

        # ── State transitions ─────────────────────────────────────────────────
        if probability >= self.entry_threshold and not in_cooldown:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        # Open episode
        if (
            not self._episode_active
            and self._consecutive_above >= self.persistence_intervals
            and not in_cooldown
        ):
            self._episode_active = True
            self._current_episode_id += 1
            self._current_episode = AlertEpisode(
                episode_id=self._current_episode_id,
                start_timestamp=str(t),
                end_timestamp=None,
                peak_probability=probability,
                peak_alert_state="WARNING",
                n_intervals=1,
                resolved=False,
            )
            logger.debug(
                f"[AlertStateMachine] Episode #{self._current_episode_id} opened "
                f"at {t} (p={probability:.3f})"
            )

        # Update active episode
        elif self._episode_active and self._current_episode is not None:
            if probability > self._current_episode.peak_probability:
                self._current_episode.peak_probability = probability
            self._current_episode.n_intervals += 1

        # Close episode
        if self._episode_active and self._current_episode is not None:
            if probability < self.exit_threshold:
                self._current_episode.end_timestamp = str(t)
                self._current_episode.resolved = True
                if probability >= self.entry_threshold * 1.5 or risk_score >= 75.0:
                    self._current_episode.peak_alert_state = "CRITICAL"
                else:
                    self._current_episode.peak_alert_state = "WARNING"
                self._episodes.append(self._current_episode)
                self._last_episode_end_ts = t
                self._episode_active = False
                self._current_episode = None
                self._consecutive_above = 0
                logger.debug(
                    f"[AlertStateMachine] Episode #{self._current_episode_id} closed at {t}"
                )

        # Assign current state
        if self._episode_active:
            if risk_score >= 75.0 or probability >= min(self.entry_threshold * 1.5, 0.90):
                self._current_state = AlertState.CRITICAL
            else:
                self._current_state = AlertState.WARNING
        elif in_cooldown:
            self._current_state = AlertState.WATCH
        elif probability >= watch_threshold:
            self._current_state = AlertState.WATCH
        else:
            self._current_state = AlertState.NORMAL

        return self._build_output(probability, t, in_cooldown)

    def _build_output(
        self,
        probability: float,
        t: pd.Timestamp,
        in_cooldown: bool,
    ) -> Dict[str, Any]:
        """Build structured alert output dict."""
        ep_info: Optional[Dict[str, Any]] = None
        if self._current_episode is not None:
            ep_info = {
                "episode_id": self._current_episode.episode_id,
                "start_timestamp": self._current_episode.start_timestamp,
                "n_intervals": self._current_episode.n_intervals,
                "peak_probability": round(self._current_episode.peak_probability, 4),
            }
        return {
            "alert_state": self._current_state.value,
            "episode_active": self._episode_active,
            "in_cooldown": in_cooldown,
            "consecutive_above_threshold": self._consecutive_above,
            "total_episodes_so_far": len(self._episodes),
            "current_episode": ep_info,
            "thresholds": {
                "entry": self.entry_threshold,
                "exit": self.exit_threshold,
                "entry_basis": self.THRESHOLD_BASIS["entry_raw" if not self.use_calibrated else "entry_calibrated"],
                "exit_basis": self.THRESHOLD_BASIS["exit_raw" if not self.use_calibrated else "exit_calibrated"],
            },
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics of all completed alert episodes.

        Episodes per day is estimated from total observations assuming 5-minute intervals.
        """
        n_ep = len(self._episodes)
        n_obs = max(self._n_total_observations, 1)
        total_hours = n_obs * 5.0 / 60.0
        episodes_per_day = round(n_ep / (total_hours / 24.0), 2) if total_hours > 0 else 0.0

        durations = []
        for ep in self._episodes:
            if ep.start_timestamp and ep.end_timestamp:
                try:
                    dur = (
                        pd.to_datetime(ep.end_timestamp)
                        - pd.to_datetime(ep.start_timestamp)
                    ).total_seconds() / 60.0
                    durations.append(dur)
                except Exception:
                    pass

        return {
            "n_completed_episodes": n_ep,
            "n_active_episode": 1 if self._episode_active else 0,
            "total_observations": n_obs,
            "observation_hours": round(total_hours, 1),
            "episodes_per_day_estimate": episodes_per_day,
            "mean_episode_duration_min": round(float(np.mean(durations)), 1) if durations else None,
            "median_episode_duration_min": round(float(np.median(durations)), 1) if durations else None,
            "method_note": (
                "Alert episodes are defined as continuous periods during which model-predicted "
                "probability exceeded the entry threshold. Hysteresis (exit_threshold) prevents "
                "re-alerting for brief dips below threshold. Cooldown (15 min) prevents "
                "consecutive episodes within one CGM interval."
            ),
        }

    def reset(self) -> None:
        """Reset state machine (e.g. for new patient session)."""
        self._current_state = AlertState.NORMAL
        self._consecutive_above = 0
        self._episode_active = False
        self._last_episode_end_ts = None
        self._current_episode = None
        self._episodes = []
        self._n_total_observations = 0
