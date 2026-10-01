r"""
GlucoTwin — Lead-Time and Feature Ablation Runner
scripts/run_lead_time_and_ablation.py

Methodologically consistent lead-time and ablation evaluation under the
canonical chronological 80/20 train/test split across all eligible HUPA-UCM cohort patients.

Guarantees:
- Zero data leakage: Feature matrices use closed='left' rolling windows.
- Strict temporal partition: First 80% of each patient's records = train, final 20% = test.
- No patient crossover between train and test windows.
- Outputs reproducible JSON reports:
    - reports/lead_time_analysis.json
    - reports/ablation_study.json

Usage:
    python scripts/run_lead_time_and_ablation.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix
from src.models.classifier import HypoClassifier
from src.evaluation.lead_time import analyze_lead_time
from src.evaluation.ablation import run_feature_ablation_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/hupa_ucm")
REPORTS_DIR = Path("reports")
HORIZON_MIN = 60
THRESHOLD_OPERATING = 0.25
TRAIN_RATIO = 0.80


def load_and_split_cohort(data_dir: Path = DATA_DIR):
    """
    Load all patient data, engineer features, and perform strict 80/20 chronological splits.
    """
    patient_files = get_patient_files(data_dir)
    logger.info(f"Discovered {len(patient_files)} patient files in {data_dir}")

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    ts_test_list = []

    for pf in patient_files:
        pid = pf.stem
        try:
            df_raw = load_patient_raw(pf)
            df_clean = preprocess_patient(df_raw, COLUMN_MAP)
            df_labeled = generate_prediction_labels(
                df_clean, "glucose_mgdl", "timestamp",
                prediction_horizon_min=HORIZON_MIN,
            )
            X, y = build_feature_matrix(df_labeled, COLUMN_MAP, label_col="label_hypo", vectorized=True)
            if len(X) < 100:
                logger.warning(f"Patient {pid} has too few samples ({len(X)}). Skipping.")
                continue

            # Strict chronological 80/20 partition per patient
            split_idx = int(len(X) * TRAIN_RATIO)
            X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
            X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]

            # Retain corresponding timestamps for lead time
            ts_col = "timestamp" if "timestamp" in df_labeled.columns else COLUMN_MAP.get("timestamp", "timestamp")
            ts_series = df_labeled.loc[X.index, ts_col]
            ts_te = ts_series.iloc[split_idx:]

            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            X_test_list.append(X_te)
            y_test_list.append(y_te)
            ts_test_list.append(ts_te)

            logger.info(f"Loaded {pid}: train={len(X_tr)}, test={len(X_te)}")
        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)
    ts_test = pd.concat(ts_test_list, ignore_index=True)

    return X_train, y_train, X_test, y_test, ts_test


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cohort data under strict chronological 80/20 split...")
    X_train, y_train, X_test, y_test, ts_test = load_and_split_cohort()
    logger.info(f"Total cohort split: Train N={len(X_train)}, Test N={len(X_test)}")

    # ── 1. Train Production Model on Chronological Train Split ──────────────
    logger.info("Training HypoClassifier on pooled chronological train split...")
    model = HypoClassifier(model_type="lightgbm", n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # ── 2. Run Lead-Time Analysis on Held-Out Test Split ─────────────────────
    logger.info("Computing lead-time and false-alarm metrics on held-out test split...")
    y_proba_test = model.predict_proba(X_test)
    lead_time_results = analyze_lead_time(
        y_true=y_test.values,
        y_pred_proba=y_proba_test,
        timestamps=ts_test,
        threshold=THRESHOLD_OPERATING,
        prediction_horizon_min=HORIZON_MIN,
    )
    lead_time_results["operating_threshold"] = THRESHOLD_OPERATING
    lead_time_results["split_protocol"] = "Chronological 80/20 holdout"
    lead_time_results["n_test_samples"] = len(X_test)

    lt_path = REPORTS_DIR / "lead_time_analysis.json"
    with open(lt_path, "w", encoding="utf-8") as f:
        json.dump(lead_time_results, f, indent=2, default=str)
    logger.info(f"Saved lead time analysis to {lt_path}")

    # ── 3. Run Feature Ablation Study ─────────────────────────────────────────
    logger.info("Running systematic feature group ablation study...")
    ablation_results = run_feature_ablation_study(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD_OPERATING,
        n_estimators=100,
        random_state=42,
    )

    abl_path = REPORTS_DIR / "ablation_study.json"
    with open(abl_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    logger.info(f"Saved ablation study to {abl_path}")

    logger.info("Execution complete.")


if __name__ == "__main__":
    main()
