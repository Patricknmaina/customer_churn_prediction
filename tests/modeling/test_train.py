# tests/modeling/test_train.py

"""
Integration test for scripts/modeling/train.py — full training pipeline.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def training_csv(tmp_path):
    """Write a 30-row synthetic CSV to tmp_path and return the path."""
    rng = np.random.RandomState(42)
    n = 30
    df = pd.DataFrame({
        "state": rng.choice(["KS", "OH", "NJ", "CA"], n),
        "account length": rng.randint(50, 200, n),
        "area code": rng.choice([408, 415, 510], n),
        "phone number": [f"555-{i:04d}" for i in range(n)],
        "international plan": rng.choice(["yes", "no"], n),
        "voice mail plan": rng.choice(["yes", "no"], n),
        "number vmail messages": rng.randint(0, 30, n),
        "total day minutes": rng.uniform(100, 300, n).round(1),
        "total day calls": rng.randint(50, 150, n),
        "total day charge": rng.uniform(15, 55, n).round(2),
        "total eve minutes": rng.uniform(50, 250, n).round(1),
        "total eve calls": rng.randint(50, 150, n),
        "total eve charge": rng.uniform(5, 25, n).round(2),
        "total night minutes": rng.uniform(100, 300, n).round(1),
        "total night calls": rng.randint(50, 150, n),
        "total night charge": rng.uniform(5, 15, n).round(2),
        "total intl minutes": rng.uniform(5, 15, n).round(1),
        "total intl calls": rng.randint(1, 10, n),
        "total intl charge": rng.uniform(1, 5, n).round(2),
        "customer service calls": rng.randint(0, 5, n),
        "churn": [False] * 22 + [True] * 8,
    })
    csv_path = tmp_path / "churn.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.slow
@pytest.mark.integration
def test_train_and_save_model_end_to_end(tmp_path, training_csv, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"

    # Patch the module-level constants in train.py
    monkeypatch.setattr("scripts.modeling.train.RAW_DATA_PATH", training_csv)
    monkeypatch.setattr("scripts.modeling.train.ARTIFACTS_DIR", artifacts_dir)

    from scripts.modeling.train import train_and_save_model

    model, metrics = train_and_save_model()

    # Verify all 6 artifact files were created
    assert (artifacts_dir / "model.joblib").exists()
    assert (artifacts_dir / "scaler.joblib").exists()
    assert (artifacts_dir / "encoder.joblib").exists()
    assert (artifacts_dir / "label_encoder.joblib").exists()
    assert (artifacts_dir / "feature_names.json").exists()
    assert (artifacts_dir / "dropped_features.json").exists()

    # Verify return values
    assert model is not None
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "recall" in metrics
