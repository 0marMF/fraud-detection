"""Explicabilidad con SHAP.

La probabilidad sola no convence a nadie en riesgo: hay que poder decir *por qué* esta
transacción huele a fraude. SHAP descompone cada predicción en cuánto empuja cada feature.
Esta lógica la usan tanto el notebook 04 como la futura API (CP5).
"""
import numpy as np
import pandas as pd
import shap

# matplotlib se importa solo dentro de las funciones que dibujan. Así, cuando la API solo
# necesita explicar una predicción (top_contributions), no arrastramos todo el stack de
# plotting al contenedor de serving.


def _frame(X, features) -> pd.DataFrame:
    # SHAP trabaja más cómodo con nombres de columna; además forzamos copia writeable.
    return pd.DataFrame(np.ascontiguousarray(np.asarray(X, dtype=float)), columns=features)


def explain(model, X, features):
    """Devuelve el objeto Explanation de SHAP para un conjunto de filas."""
    explainer = shap.TreeExplainer(model)
    return explainer(_frame(X, features))


def summary_figure(model, X, features, out_path, max_display=15):
    """Beeswarm global: qué features mueven la predicción y en qué dirección."""
    import matplotlib
    matplotlib.use("Agg")            # sin ventana: solo guardamos figuras
    import matplotlib.pyplot as plt
    sv = explain(model, X, features)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    fig = plt.gcf(); fig.set_size_inches(10, 7)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return sv


def waterfall_figure(explanation, idx, out_path):
    """Explicación de UNA transacción concreta (qué la empujó a fraude o a legítima)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    shap.plots.waterfall(explanation[idx], max_display=12, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def top_contributions(model, x_row, features, k=5):
    """Top-k features que más pesaron en una predicción individual.

    Pensada para la API/dashboard: recibe una fila (ya escalada, en el orden de `features`)
    y devuelve [(feature, valor_shap), ...] ordenado por impacto absoluto.
    """
    sv = explain(model, [np.asarray(x_row, dtype=float)], features)
    valores = sv.values[0]
    orden = np.argsort(np.abs(valores))[::-1][:k]
    return [(features[i], float(valores[i])) for i in orden]
