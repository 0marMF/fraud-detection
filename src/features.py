"""Feature engineering, split y balanceo.

Aquí está la parte delicada del proyecto: el orden importa. Primero partimos train/test,
LUEGO ajustamos el scaler (solo con train) y aplicamos SMOTE (solo a train). Si lo hiciéramos
al revés estaríamos filtrando información del test hacia el modelo (data leakage).
"""
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import load_config, path


def engineer(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Crea Hour, Amount_log e Is_night y descarta las columnas originales."""
    df = df.copy()
    df["Hour"] = (df["Time"] / 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])          # Amount está muy sesgado; log lo calma
    ini, fin = cfg["features"]["night_start"], cfg["features"]["night_end"]
    # El EDA mostró que la TASA de fraude sube de madrugada, así que marcamos esa franja.
    df["Is_night"] = df["Hour"].apply(lambda h: 1 if (h >= ini or h <= fin) else 0)
    return df.drop(columns=["Time", "Amount"])


def make_splits(df: pd.DataFrame, cfg: dict, save: bool = True) -> dict:
    """Devuelve los arrays listos para modelar (train balanceado, test intacto)."""
    seed = cfg["seed"]
    X, y = df.drop(columns=["Class"]), df["Class"]
    features = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed
    )

    cols = cfg["features"]["scale_cols"]
    scaler = RobustScaler()                            # robusto a outliers, que aquí sobran
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[cols] = scaler.fit_transform(X_tr[cols])      # fit SOLO con train
    X_te[cols] = scaler.transform(X_te[cols])

    # SMOTE solo en train. En test sería inventar fraudes que nunca ocurrieron.
    X_tr_res, y_tr_res = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)

    splits = {
        "X_train": X_tr_res, "X_test": X_te,
        "y_train": y_tr_res, "y_test": y_te,
        # Guardamos también el train SIN SMOTE (ya escalado). Lo necesita la validación
        # cruzada: el SMOTE hay que aplicarlo DENTRO de cada fold, no antes, o filtra.
        "X_train_raw": X_tr, "y_train_raw": y_tr,
        "features": features, "scaler": scaler, "cols_scaled": cols,
    }
    if save:
        with open(path(cfg["data"]["splits"]), "wb") as f:
            pickle.dump(splits, f)
    return splits
