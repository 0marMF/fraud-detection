"""Métricas y traducción a impacto de negocio.

Con un 0.17% de fraude, accuracy no sirve de nada. Las métricas honestas son PR-AUC, F1,
recall y precision sobre la clase fraude, siempre medidas en el test real (sin SMOTE).
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _writeable(a) -> np.ndarray:
    # Arrays que vienen de un DataFrame pickleado llegan como solo-lectura y sklearn 1.5 +
    # numpy 2 revientan al intentar marcarlos writeable. Una copia contigua lo soluciona.
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    # zero_division=0: si no hay positivos predichos (p.ej. el baseline tonto) no queremos
    # un warning, queremos un 0 limpio.
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def dummy_baseline(X_train, y_train, X_test, y_test) -> dict:
    """Baseline tonto: predecir siempre la clase mayoritaria.

    No es para usarlo, es la vara de medir: si nuestro modelo no le saca ventaja clara,
    no está aportando nada. Con 0.17% de fraude, su recall y PR-AUC se van casi a cero.
    """
    X_train, X_test = _writeable(X_train), _writeable(X_test)
    y_train = np.asarray(y_train).ravel().astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    proba = dummy.predict_proba(X_test)[:, 1]
    return classification_metrics(y_test, proba, 0.5)


def cross_val_prauc(model, X_raw, y_raw, seed, n_splits=5) -> tuple[float, float]:
    """PR-AUC por validación cruzada, con SMOTE DENTRO de cada fold (sin fuga).

    Aplicar SMOTE antes de partir en folds contamina la validación. Lo metemos en un
    Pipeline de imblearn para que se reajuste en cada fold de entrenamiento.
    """
    X_raw = _writeable(X_raw)
    y_raw = np.asarray(y_raw).ravel().astype(int)
    pipe = ImbPipeline([("smote", SMOTE(random_state=seed)), ("clf", clone(model))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X_raw, y_raw, cv=skf,
                             scoring="average_precision", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def threshold_cost_curve(y_true, proba, amount, fp_cost):
    """Coste total en euros para cada umbral posible.

    La idea de negocio: un fraude que se escapa (FN) cuesta el monto real de esa
    transacción; una falsa alarma (FP) cuesta lo que vale revisarla a mano (fp_cost).
    El mejor umbral es el que minimiza la suma de ambos, no el 0.5 por defecto.
    """
    y_true, proba, amount = np.asarray(y_true), np.asarray(proba), np.asarray(amount)
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn = (y_true == 1) & (~pred)            # fraude no detectado -> perdemos el monto
        fp = (y_true == 0) & (pred)             # falsa alarma -> coste de revisión
        costs.append(float(amount[fn].sum() + fp_cost * fp.sum()))
    costs = np.array(costs)
    best = int(np.argmin(costs))
    return thresholds, costs, float(thresholds[best]), float(costs[best])


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
