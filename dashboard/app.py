"""
🛡️ Dashboard de Detección de Fraude — Credit Card Fraud Detection
Omar Mora Flores

Interfaz Streamlit que carga el modelo XGBoost entrenado (Fase 3) y predice en vivo si una
transacción es fraudulenta a partir de unas pocas features clave.

Ejecutar:  streamlit run dashboard/app.py
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Localización de rutas (robusta: corre desde la raíz o desde dashboard/)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "best_model.pkl").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
MODEL_PATH = ROOT / "src" / "best_model.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS = ROOT / "reports"


# --------------------------------------------------------------------------------------
# Lógica pura (testeable sin Streamlit)
# --------------------------------------------------------------------------------------
def load_artifact(path=MODEL_PATH):
    """Carga el modelo y sus metadatos (features, scaler, columnas escaladas)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(path=METRICS_PATH):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_feature_vector(art, amount, hour, v_values):
    """Construye el vector de features en el orden correcto y aplica el escalado.

    amount   : monto en euros (se transforma a Amount_log)
    hour     : hora del día 0-24 (deriva Is_night)
    v_values : dict {nombre_feature: valor} para las V-features expuestas en la UI
    """
    features = art["features"]
    scaler = art["scaler"]
    cols_scaled = art["cols_scaled"]

    row = {f: 0.0 for f in features}          # V-features no expuestas → 0 (media PCA)
    row.update({k: float(v) for k, v in v_values.items() if k in row})
    if "Amount_log" in row:
        row["Amount_log"] = float(np.log1p(max(amount, 0)))
    if "Hour" in row:
        row["Hour"] = float(hour) % 24
    if "Is_night" in row:
        row["Is_night"] = 1.0 if (hour >= 22 or hour <= 6) else 0.0

    X = np.array([[row[f] for f in features]], dtype=np.float64)
    idx = [features.index(c) for c in cols_scaled]
    # Pasar un DataFrame con los nombres con que se ajustó el scaler (evita el UserWarning)
    X[:, idx] = scaler.transform(pd.DataFrame(X[:, idx], columns=cols_scaled))
    return X


def predict(art, X):
    """Devuelve (label, probabilidad_de_fraude)."""
    proba = float(art["model"].predict_proba(X)[0, 1])
    return int(proba >= 0.5), proba


def top_features(art, k=6):
    """Top-k V-features más importantes según el modelo (para exponer en la UI)."""
    imp = pd.Series(art["model"].feature_importances_, index=art["features"])
    v_only = imp[[f for f in art["features"] if f.startswith("V")]]
    return v_only.sort_values(ascending=False).head(k).index.tolist()


# --------------------------------------------------------------------------------------
# Interfaz Streamlit
# --------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Detección de Fraude", page_icon="🛡️", layout="wide")

    st.title("🛡️ Detección de Fraude en Transacciones")
    st.caption("Modelo XGBoost · ingresa los datos de una transacción y obtén la predicción")

    art = _load_artifact_cached()
    metrics = load_metrics()
    tops = top_features(art, k=6)

    # ---------------- Sidebar: inputs ----------------
    st.sidebar.header("📝 Datos de la transacción")
    amount = st.sidebar.number_input("Monto (€)", min_value=0.0, value=100.0, step=10.0)
    hour = st.sidebar.slider("Hora del día", 0, 24, 3,
                             help="El fraude tiende a concentrarse de madrugada (2–4 a.m.)")
    st.sidebar.markdown("**Componentes PCA principales** *(V-features)*")
    v_values = {}
    for f in tops:
        v_values[f] = st.sidebar.slider(f, -20.0, 10.0, 0.0, 0.1)
    predecir = st.sidebar.button("🔍 Predecir", use_container_width=True)

    # ---------------- Panel principal ----------------
    col1, col2 = st.columns([1, 1])

    if predecir:
        X = build_feature_vector(art, amount, hour, v_values)
        label, proba = predict(art, X)
        with col1:
            if label == 1:
                st.error("## 🚨 FRAUDE DETECTADO")
            else:
                st.success("## ✅ TRANSACCIÓN LEGÍTIMA")
            st.metric("Probabilidad de fraude", f"{proba*100:.2f}%")
            st.progress(min(proba, 1.0))
            es_noche = "Sí" if (hour >= 22 or hour <= 6) else "No"
            st.caption(f"Monto: €{amount:,.2f} · Hora: {hour}:00 · Nocturna: {es_noche}")
    else:
        with col1:
            st.info("Ajusta los valores en la barra lateral y pulsa **Predecir**.")

    # ---------------- Importancia de features ----------------
    with col2:
        st.subheader("📊 Importancia de features del modelo")
        imp = (pd.Series(art["model"].feature_importances_, index=art["features"])
               .sort_values(ascending=False).head(12))
        st.bar_chart(imp)

    # ---------------- Métricas del modelo ----------------
    if metrics:
        st.divider()
        st.subheader(f"📈 Rendimiento del modelo ({metrics['best_model']})")
        b = metrics["best"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{b['roc_auc']:.3f}")
        m2.metric("PR-AUC", f"{b['pr_auc']:.3f}")
        m3.metric("F1 (fraude)", f"{b['f1']:.3f}")
        m4.metric("Recall (fraude)", f"{b['recall']*100:.0f}%")
        st.caption(
            f"Detecta {b['tp']} de {b['frauds_total_test']} fraudes del test real con "
            f"{b['fp']} falsas alarmas · ~€{b['losses_avoided_eur']:,.0f} en pérdidas evitadas."
        )

    # ---------------- Visualizaciones del EDA ----------------
    st.divider()
    st.subheader("🔬 Análisis exploratorio (EDA)")
    eda_imgs = {
        "Distribución de clases": REPORTS / "01_class_distribution.png",
        "Análisis temporal": REPORTS / "03_temporal_analysis.png",
        "Correlación de features": REPORTS / "04_feature_correlation.png",
    }
    tabs = st.tabs(list(eda_imgs.keys()))
    for tab, (titulo, ruta) in zip(tabs, eda_imgs.items()):
        with tab:
            if ruta.exists():
                st.image(str(ruta), use_column_width=True)
            else:
                st.warning(f"Imagen no encontrada: {ruta.name}")


@st.cache_resource
def _load_artifact_cached():
    try:
        return load_artifact()
    except FileNotFoundError:
        st.error(
            "No se encontró `src/best_model.pkl`. Ejecuta primero los notebooks "
            "01 → 02 → 03 para entrenar y serializar el modelo."
        )
        st.stop()


if __name__ == "__main__":
    main()
