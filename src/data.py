"""Carga y limpieza del dataset crudo."""
import pandas as pd

from .config import load_config, path


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(path(cfg["data"]["raw_csv"]))


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Quita duplicados exactos.

    En el EDA salieron 1.081 filas duplicadas. Hay que eliminarlas ANTES del split: si no,
    una misma fila puede acabar a la vez en train y en test y nos infla las métricas sin que
    nos demos cuenta. Devuelvo también cuántas quité, para poder reportarlo.
    """
    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, antes - len(df)
