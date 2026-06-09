"""Tests de métricas, curva de coste y reconstrucción del monto."""
import numpy as np

from src.evaluate import (classification_metrics, reconstruct_amount,
                          threshold_cost_curve)
from src.features import engineer, make_splits


def test_metricas_en_rango():
    y = [0, 1, 0, 1, 0]
    p = [0.1, 0.9, 0.2, 0.7, 0.3]
    m = classification_metrics(y, p, 0.5)
    for k in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
        assert 0.0 <= m[k] <= 1.0


def test_curva_coste_elige_minimo():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    amount = np.array([10.0, 10.0, 100.0, 100.0])
    ths, costs, t_star, cost_star = threshold_cost_curve(y, p, amount, fp_cost=1.0)
    assert 0.0 < t_star < 1.0
    assert cost_star == costs.min()           # el umbral elegido es el de menor coste
    assert len(ths) == len(costs)


def test_reconstruct_amount_no_negativo(synth_df, cfg):
    df = engineer(synth_df, cfg)
    s = make_splits(df, cfg, save=False)
    amount = reconstruct_amount(s["X_test"], s["scaler"], s["cols_scaled"], s["features"])
    # El monto original era >= 0; tras escalar y deshacer debe seguir siéndolo.
    assert (amount >= -1e-6).all()
    assert len(amount) == len(s["y_test"])
