"""Entrenamiento, selección y serialización del modelo."""
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import path
from .evaluate import classification_metrics


def _as_array(a) -> np.ndarray:
    # sklearn 1.5 + numpy 2 a veces se queja de arrays de solo lectura (vienen del pickle de
    # pandas). Forzamos una copia writeable y contigua para evitar ese error tonto.
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def build_models(cfg: dict) -> dict:
    seed = cfg["seed"]
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_jobs=-1, random_state=seed, **cfg["model"]["random_forest"]
        ),
        "XGBoost": XGBClassifier(
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed,
            **cfg["model"]["xgboost"]
        ),
    }


def train_and_select(splits: dict, cfg: dict) -> dict:
    """Entrena los tres modelos y elige el mejor por PR-AUC (la métrica más fiable aquí)."""
    X_tr, X_te = _as_array(splits["X_train"]), _as_array(splits["X_test"])
    y_tr = np.asarray(splits["y_train"]).ravel().astype(int)
    y_te = np.asarray(splits["y_test"]).ravel().astype(int)

    fitted, results, probas = {}, {}, {}
    for name, model in build_models(cfg).items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        fitted[name], probas[name] = model, proba
        results[name] = classification_metrics(y_te, proba, cfg["threshold"])

    best = max(results, key=lambda k: results[k]["pr_auc"])
    return {"fitted": fitted, "results": results, "probas": probas, "best": best,
            "y_test": y_te}


def save_model(model, name: str, splits: dict, cfg: dict) -> None:
    """Guarda el modelo junto a lo que el dashboard/API necesita para reconstruir el input."""
    artefacto = {
        "model": model, "model_name": name,
        "features": splits["features"], "scaler": splits["scaler"],
        "cols_scaled": splits["cols_scaled"],
    }
    with open(path(cfg["paths"]["model"]), "wb") as f:
        pickle.dump(artefacto, f)
