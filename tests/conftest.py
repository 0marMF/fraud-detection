"""Fixtures compartidas para los tests.

Clave: los tests NO dependen del dataset real (que no se versiona). Trabajan con datos
sintéticos que imitan el esquema de creditcard.csv, así corren en cualquier sitio (incluido CI)
en segundos.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cfg():
    """Config mínima equivalente a config.yaml, con modelos pequeños para que el test vuele."""
    return {
        "seed": 42,
        "split": {"test_size": 0.25},
        "features": {"scale_cols": ["Amount_log", "Hour"], "night_start": 22, "night_end": 6},
        "model": {
            "random_forest": {"n_estimators": 20},
            "xgboost": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2,
                        "subsample": 1.0, "colsample_bytree": 1.0},
        },
        "threshold": 0.5,
        "costs": {"fp_review_cost": 3.0},
    }


@pytest.fixture
def synth_df():
    """DataFrame con el mismo esquema que creditcard.csv: Time, V1..V28, Amount, Class.

    400 filas, 40 fraudes (suficientes para que SMOTE encuentre vecinos).
    """
    rng = np.random.RandomState(0)
    n = 400
    data = {f"V{i}": rng.randn(n) for i in range(1, 29)}
    data["Time"] = rng.randint(0, 172_792, n).astype(float)
    data["Amount"] = np.abs(rng.randn(n)) * 50.0
    y = np.zeros(n, dtype=int)
    y[:40] = 1
    rng.shuffle(y)
    data["Class"] = y
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return pd.DataFrame(data)[cols]
