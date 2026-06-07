"""Métricas y traducción a impacto de negocio.

Con un 0.17% de fraude, accuracy no sirve de nada. Las métricas honestas son PR-AUC, F1,
recall y precision sobre la clase fraude, siempre medidas en el test real (sin SMOTE).
"""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
    }


def reconstruct_amount(X_test, scaler, cols_scaled, features) -> np.ndarray:
    """Recupera el monto real en euros de cada transacción de test.

    En el split guardamos Amount_log escalado, no el Amount original. Para hablar de dinero
    deshacemos el escalado (inverse_transform) y el log1p (expm1). Así el impacto de negocio
    sale en euros de verdad, no en una escala abstracta.
    """
    idx = [features.index(c) for c in cols_scaled]
    arr = np.ascontiguousarray(np.asarray(X_test)[:, idx], dtype=float)
    unscaled = scaler.inverse_transform(arr)
    return np.expm1(unscaled[:, cols_scaled.index("Amount_log")])


def business_impact(y_true, pred, amount) -> dict:
    """Cuánto dinero salvamos y cuánto se nos escapa, según la matriz de confusión."""
    y_true, pred, amount = np.asarray(y_true), np.asarray(pred), np.asarray(amount)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    es_fraude = y_true == 1
    return {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "frauds_total": int(es_fraude.sum()),
        "losses_avoided_eur": float(amount[es_fraude & (pred == 1)].sum()),
        "losses_missed_eur": float(amount[es_fraude & (pred == 0)].sum()),
    }
