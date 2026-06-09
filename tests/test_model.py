"""Tests del modelado: que se construyan los 3 modelos y que las probabilidades sean válidas."""
import numpy as np

from src.features import engineer, make_splits
from src.model import build_models, train_and_select


def test_build_models_tres(cfg):
    assert set(build_models(cfg)) == {"Logistic Regression", "Random Forest", "XGBoost"}


def test_train_and_select(synth_df, cfg):
    df = engineer(synth_df, cfg)
    s = make_splits(df, cfg, save=False)
    out = train_and_select(s, cfg)

    assert out["best"] in build_models(cfg)
    # las probabilidades deben estar en [0, 1] para los tres modelos
    for nombre, proba in out["probas"].items():
        proba = np.asarray(proba)
        assert ((proba >= 0) & (proba <= 1)).all(), nombre
    # hay una métrica por modelo
    assert set(out["results"]) == set(build_models(cfg))
