"""Tests de la lógica de scoring que consume la API (src/score.py)."""
import numpy as np

from src import score
from src.features import engineer, make_splits
from src.model import build_models


def _artefacto(synth_df, cfg):
    """Arma un artefacto como el que guarda el pipeline, pero entrenado en sintético."""
    df = engineer(synth_df, cfg)
    s = make_splits(df, cfg, save=False)
    model = build_models(cfg)["XGBoost"]
    model.fit(np.ascontiguousarray(np.asarray(s["X_train"], float)),
              np.asarray(s["y_train"]).astype(int))
    return {"model": model, "model_name": "XGBoost", "features": s["features"],
            "scaler": s["scaler"], "cols_scaled": s["cols_scaled"], "threshold": 0.3}


def test_build_features_shape(synth_df, cfg):
    art = _artefacto(synth_df, cfg)
    X = score.build_features(art, amount=50.0, hour=14, v_values={})
    assert X.shape == (1, len(art["features"]))


def test_predict_devuelve_veredicto_explicado(synth_df, cfg):
    art = _artefacto(synth_df, cfg)
    out = score.predict(art, amount=120.0, hour=3, v_values={"V1": -2.0})
    assert 0.0 <= out["probability"] <= 1.0
    assert isinstance(out["is_fraud"], bool)
    assert out["threshold"] == 0.3                 # usa el umbral guardado, no 0.5
    assert len(out["top_features"]) == 5           # explicación SHAP
    assert all({"feature", "shap"} <= set(t) for t in out["top_features"])
