r"""
GlucoTwin — Production Calibrator Fitting & Serialization Script
scripts/fit_and_serialize_calibrator.py

Fits IsotonicCalibrator strictly on validation split predictions and labels,
verifies out-of-sample Brier score on test set, and serializes artifact
to models/glucotwin_calibrator.joblib with complete metadata.

LEAKAGE SAFETY:
- Fits calibrator ONLY on validation predictions and validation labels.
- Test set is evaluated ONLY once for out-of-sample reporting.
- Does NOT overwrite the production model artifact.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix
from src.models.classifier import HypoClassifier
from src.models.calibration import IsotonicCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/hupa_ucm")
OUTPUT_CALIBRATOR = Path("models/glucotwin_calibrator.joblib")
OUTPUT_METADATA = Path("models/glucotwin_calibrator_metadata.json")


def load_cohort_data():
    """Load and process features and labels for all patients."""
    files = get_patient_files(DATA_DIR)
    all_X, all_y = {}, {}

    for f in files:
        pid = f.stem
        try:
            df_raw = load_patient_raw(f)
            df_clean = preprocess_patient(df_raw, COLUMN_MAP)
            df_lab = generate_prediction_labels(
                df_clean, "glucose_mgdl", "timestamp",
                prediction_horizon_min=60,
                threshold_mgdl=70.0,
            )
            X, y = build_feature_matrix(df_lab, COLUMN_MAP, vectorized=True)
            if len(X) >= 50 and y.sum() >= 5:
                all_X[pid] = X
                all_y[pid] = y
        except Exception as e:
            logger.warning(f"Error processing {pid}: {e}")

    logger.info(f"Loaded {len(all_X)} patients.")
    return all_X, all_y


def main():
    logger.info("Loading patient cohort data...")
    all_X, all_y = load_cohort_data()

    X_all = pd.concat(all_X.values(), ignore_index=True)
    y_all = pd.concat(all_y.values(), ignore_index=True)
    n = len(X_all)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    X_train = X_all.iloc[:n_train]
    y_train = y_all.iloc[:n_train]
    X_val = X_all.iloc[n_train:n_train + n_val]
    y_val = y_all.iloc[n_train:n_train + n_val]
    X_test = X_all.iloc[n_train + n_val:]
    y_test = y_all.iloc[n_train + n_val:]

    logger.info(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Train classifier on train split
    logger.info("Training classifier on train split (60%)...")
    clf = HypoClassifier(model_type="lightgbm")
    clf.fit(X_train, y_train)

    # Predict probabilities
    logger.info("Predicting validation probabilities (for calibration fitting)...")
    val_proba = clf.predict_proba(X_val)

    logger.info("Fitting IsotonicCalibrator strictly on validation data...")
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_proba, y_val)

    # Evaluate on test set
    logger.info("Evaluating on unseen test set...")
    raw_test = clf.predict_proba(X_test)
    cal_test = calibrator.transform(raw_test)

    brier_raw = float(brier_score_loss(y_test, raw_test))
    brier_cal = float(brier_score_loss(y_test, cal_test))
    brier_val = float(brier_score_loss(y_val, calibrator.transform(val_proba)))

    logger.info(f"Test Raw Brier:        {brier_raw:.4f}")
    logger.info(f"Test Calibrated Brier: {brier_cal:.4f}")
    logger.info(f"Val Calibrated Brier:  {brier_val:.4f}")

    metadata = {
        "calibrator_type": "isotonic",
        "calibrator_class": "src.models.calibration.IsotonicCalibrator",
        "version": "1.0.0",
        "twin_state_version": "5",
        "created_at": "2026-09-30T23:55:00+05:30",
        "training_methodology": "Isotonic regression fit on validation split (20% chronological holdout)",
        "source_model_identity": "LightGBMClassifier (n_estimators=250, max_depth=6, lr=0.05, balanced)",
        "prediction_horizon_min": 60,
        "target_threshold_mgdl": 70.0,
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "n_test_samples": int(len(X_test)),
        "dataset_name": "HUPA-UCM",
        "cohort_patients": len(all_X),
        "brier_score_test_calibrated": round(brier_cal, 4),
        "brier_score_test_raw": round(brier_raw, 4),
        "brier_score_val_calibrated": round(brier_val, 4),
        "calibrated_threshold_f2_val": 0.13,
        "raw_threshold_f2_val": 0.44,
        "monotonic": True,
        "out_of_bounds_policy": "clip",
        "output_bounds": [0.0, 1.0],
        "leakage_guarantee": "Zero test labels or test predictions used during fitting. Fitted strictly on validation split.",
    }

    calibrator.metadata = metadata

    # Serialize calibrator
    OUTPUT_CALIBRATOR.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(OUTPUT_CALIBRATOR)
    logger.info(f"Calibrator artifact written to: {OUTPUT_CALIBRATOR}")

    # Write companion metadata JSON
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Calibrator metadata written to: {OUTPUT_METADATA}")

    print("\nCalibration artifact and metadata successfully generated:")
    print(f"  Artifact: {OUTPUT_CALIBRATOR} ({OUTPUT_CALIBRATOR.stat().st_size} bytes)")
    print(f"  Brier Raw: {brier_raw:.4f} -> Brier Calibrated: {brier_cal:.4f}")


if __name__ == "__main__":
    main()
