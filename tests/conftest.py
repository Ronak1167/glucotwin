"""
GlucoTwin — pytest configuration and session fixture setup
tests/conftest.py
"""

from pathlib import Path
import pytest
from scripts.generate_ci_fixtures import generate_fixtures


@pytest.fixture(scope="session", autouse=True)
def ensure_hupa_fixtures():
    """Ensure data/raw/hupa_ucm has patient fixtures before running tests."""
    data_dir = Path("data/raw/hupa_ucm")
    existing = list(data_dir.rglob("HUPA*.csv")) if data_dir.exists() else []
    if len(existing) < 25:
        generate_fixtures(target_dir=data_dir / "Preprocessed", n_patients=25, n_rows=1200)
