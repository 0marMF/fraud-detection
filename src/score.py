"""Scoring de una transacción: del input crudo a un veredicto explicado.

Una sola fuente de verdad para puntuar: la usan tanto la API (`src/api.py`) como el dashboard.
Recibe lo que un humano sabe de la transacción (monto, hora y las componentes V) y devuelve la
probabilidad, el veredicto según el umbral por coste, y las features que más pesaron.
"""
import pickle

import numpy as np
import pandas as pd

from . import explain
from .config import load_config, path


def load_artifact(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    with open(path(cfg["paths"]["model"]), "rb") as f:
        return pickle.load(f)


def build_features(art: dict, amount: float, hour: float, v_values: dict | None) -> np.ndarray:
    """Arma el vector en el orden que espera el modelo y escala lo que toca.

    Las V que no nos pasen se quedan en 0 (su media, porque vienen de PCA centrado). De `amount`
    derivamos `Amount_log`; de `hour`, `Is_night`. Luego escalamos solo Amount_log y Hour con el
    mismo scaler que se entrenó.
    """
    features, scaler, cols = art["features"], art["scaler"], art["cols_scaled"]
    row = {f: 0.0 for f in features}
    for k, v in (v_values or {}).items():
        if k in row:
            row[k] = float(v)
    if "Amount_log" in row:
        row["Amount_log"] = float(np.log1p(max(amount, 0.0)))
    if "Hour" in row:
        row["Hour"] = float(hour) % 24
    if "Is_night" in row:
        row["Is_night"] = 1.0 if (hour >= 22 or hour <= 6) else 0.0

    X = np.array([[row[f] for f in features]], dtype=np.float64)
    idx = [features.index(c) for c in cols]
    # DataFrame con nombres para que el scaler no se queje de "feature names".
    X[:, idx] = scaler.transform(pd.DataFrame(X[:, idx], columns=cols))
    return X


def predict(art: dict, amount: float, hour: float, v_values: dict | None = None,
            with_explanation: bool = True) -> dict:
    X = build_features(art, amount, hour, v_values)
    proba = float(art["model"].predict_proba(X)[0, 1])
    threshold = float(art.get("threshold", 0.5))
    resultado = {
        "probability": round(proba, 4),
        "is_fraud": bool(proba >= threshold),
        "threshold": round(threshold, 4),
    }
    if with_explanation:
        resultado["top_features"] = [
            {"feature": f, "shap": round(v, 4)}
            for f, v in explain.top_contributions(art["model"], X[0], art["features"], k=5)
        ]
    return resultado
