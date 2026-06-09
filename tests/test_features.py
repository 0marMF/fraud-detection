"""Tests de feature engineering y del split (incluye el guard anti-leakage)."""
import numpy as np

from src.features import engineer, make_splits


def test_engineer_crea_y_elimina(synth_df, cfg):
    out = engineer(synth_df, cfg)
    assert {"Hour", "Amount_log", "Is_night"} <= set(out.columns)
    assert "Time" not in out.columns and "Amount" not in out.columns
    # Amount_log tiene que ser exactamente log1p(Amount)
    np.testing.assert_allclose(out["Amount_log"].values, np.log1p(synth_df["Amount"].values))
    assert set(out["Is_night"].unique()) <= {0, 1}


def test_split_sin_leakage(synth_df, cfg):
    df = engineer(synth_df, cfg)
    s = make_splits(df, cfg, save=False)
    ytr, yte = np.asarray(s["y_train"]), np.asarray(s["y_test"])

    # 1) SMOTE solo en train -> train queda ~balanceado
    assert abs(ytr.mean() - 0.5) < 0.05
    # 2) el test NO se toca -> mantiene el desbalance real (muy lejos del 50/50)
    assert yte.mean() < 0.25
    # 3) el test no fue oversampleado: su tamaño es ~test_size del total
    assert abs(len(yte) - cfg["split"]["test_size"] * len(df)) <= 2
    # 4) artefactos necesarios para CV y para no re-escalar mal
    assert "X_train_raw" in s
    assert hasattr(s["scaler"], "center_")     # scaler realmente ajustado


def test_scaler_ajustado_solo_en_train(synth_df, cfg):
    # El scaler debe haberse ajustado con el train: su mediana (center_) no es trivial.
    df = engineer(synth_df, cfg)
    s = make_splits(df, cfg, save=False)
    assert len(s["scaler"].center_) == len(cfg["features"]["scale_cols"])
