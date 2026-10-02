"""
GlucoTwin — CI Fixture Generator
scripts/generate_ci_fixtures.py

Generates lightweight, self-contained synthetic HUPA-UCM patient CSV fixtures
when running in CI environments where raw clinical data is not checked into git.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_fixtures(target_dir: Path = Path("data/raw/hupa_ucm/Preprocessed"), n_patients: int = 25, n_rows: int = 1200):
    existing = list(Path("data/raw/hupa_ucm").rglob("HUPA*.csv")) if Path("data/raw/hupa_ucm").exists() else []
    if len(existing) >= n_patients:
        print(f"[CI Fixtures] Found {len(existing)} existing files in data/raw/hupa_ucm. Skipping generation.")
        return

    print(f"[CI Fixtures] Generating {n_patients} synthetic patient files in {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    base_time = pd.Timestamp("2022-01-01 08:00:00")

    for p_idx in range(1, n_patients + 1):
        pid = f"HUPA{p_idx:04d}P"
        filepath = target_dir / f"{pid}.csv"

        times = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n_rows)]
        # Generate plausible glucose walk with some hypoglycemia episodes
        glucose = np.zeros(n_rows)
        glucose[0] = 120.0 + np.random.uniform(-20, 20)
        for i in range(1, n_rows):
            drift = np.random.normal(0, 3.0)
            # Add occasional meal or insulin pull
            if i % 150 == 0:
                drift -= 30.0  # drop into hypo
            elif i % 150 == 25:
                drift += 40.0  # rebound
            glucose[i] = np.clip(glucose[i - 1] + drift, 45.0, 350.0)

        df = pd.DataFrame({
            "time": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "glucose": np.round(glucose, 1),
            "calories": np.round(np.random.uniform(1.0, 10.0, n_rows), 2),
            "heart_rate": np.round(np.random.uniform(60.0, 100.0, n_rows), 1),
            "steps": np.random.choice([0.0, 10.0, 35.0, 80.0], size=n_rows),
            "basal_rate": np.round(np.random.uniform(0.05, 0.15, n_rows), 4),
            "bolus_volume_delivered": np.where(np.random.rand(n_rows) < 0.05, np.round(np.random.uniform(1.0, 5.0, n_rows), 2), 0.0),
            "carb_input": np.where(np.random.rand(n_rows) < 0.04, np.round(np.random.uniform(15.0, 60.0, n_rows), 1), 0.0),
        })

        df.to_csv(filepath, sep=";", index=False)

    print(f"[CI Fixtures] Successfully created {n_patients} synthetic patient fixtures.")


if __name__ == "__main__":
    generate_fixtures()
