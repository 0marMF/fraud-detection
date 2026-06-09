"""Tests de carga y limpieza."""
import pandas as pd

from src.data import clean


def test_clean_quita_duplicados(synth_df):
    # Metemos 5 filas repetidas a propósito; clean debe quitarlas exactamente.
    con_dups = pd.concat([synth_df, synth_df.iloc[:5]], ignore_index=True)
    limpio, n_quitados = clean(con_dups)
    assert n_quitados == 5
    assert limpio.duplicated().sum() == 0


def test_clean_no_inventa_filas(synth_df):
    limpio, n = clean(synth_df)
    assert n == 0                      # el sintético no trae duplicados
    assert len(limpio) == len(synth_df)
    assert set(limpio["Class"].unique()) <= {0, 1}
