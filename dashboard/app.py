"""
Dashboard de detección de fraude (Streamlit).

Ejecutar:  streamlit run dashboard/app.py

El dashboard NO reimplementa el scoring: se lo pide a la API (`src/api.py`). Si la API no está
levantada, cae a un modo local que usa exactamente la misma función (`src.score.predict`), así
siempre funciona al clonar el repo. La URL de la API se configura con la variable FRAUD_API_URL.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Raíz del proyecto en el path para poder importar src/ (fallback local de scoring).
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "best_model.pkl").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src import score  # noqa: E402

API_URL = os.environ.get("FRAUD_API_URL", "http://localhost:8000")
REPORTS = ROOT / "reports"
METRICS_PATH = REPORTS / "metrics.json"


@st.cache_resource
def get_artifact():
    return score.load_artifact()


def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def score_transaction(amount, hour, v_values):
    """Intenta la API; si no contesta, puntúa en local con la misma lógica."""
    payload = {"amount": amount, "hour": hour, "v_features": v_values}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
        if r.ok:
            return r.json(), "API"
    except requests.RequestException:
        pass
    return score.predict(get_artifact(), amount, hour, v_values), "local"


def top_v_features(art, k=6):
    """V-features con más peso en el modelo, para exponerlas como sliders."""
    imp = pd.Series(art["model"].feature_importances_, index=art["features"])
    solo_v = imp[[f for f in art["features"] if f.startswith("V")]]
    return solo_v.sort_values(ascending=False).head(k).index.tolist()


def main():
    st.set_page_config(page_title="Deteccion de Fraude", layout="wide")
    st.title("Detección de fraude en transacciones")
    st.caption("Modelo XGBoost servido por una API · introduce una transacción y obtén el veredicto")

    art = get_artifact()
    metrics = load_metrics()

    # ---------------- Sidebar: inputs ----------------
    st.sidebar.header("Datos de la transacción")
    amount = st.sidebar.number_input("Monto (EUR)", min_value=0.0, value=100.0, step=10.0)
    hour = st.sidebar.slider("Hora del día", 0, 24, 3,
                             help="El fraude se concentra de madrugada (2-4 a.m.)")
    st.sidebar.markdown("**Componentes PCA principales (V-features)**")
    v_values = {f: st.sidebar.slider(f, -20.0, 10.0, 0.0, 0.1) for f in top_v_features(art)}

    thr_default = float(art.get("threshold", 0.5))
    threshold = st.sidebar.slider(
        "Umbral de decisión", 0.01, 0.99, thr_default, 0.01,
        help="Arranca en el umbral óptimo por coste. Súbelo para menos falsas alarmas; "
             "bájalo para atrapar más fraude (más recall).")
    predecir = st.sidebar.button("Predecir", use_container_width=True)

    col1, col2 = st.columns(2)

    if predecir:
        res, modo = score_transaction(amount, hour, v_values)
        proba = res["probability"]
        es_fraude = proba >= threshold            # el veredicto usa el umbral elegido en la UI
        with col1:
            if es_fraude:
                st.error("## FRAUDE DETECTADO")
            else:
                st.success("## Transacción legítima")
            st.metric("Probabilidad de fraude", f"{proba*100:.2f}%")
            st.progress(min(proba, 1.0))
            nota = " (óptimo por coste)" if abs(threshold - thr_default) < 1e-9 else ""
            st.caption(f"Umbral aplicado: {threshold:.2f}{nota} · scoring vía: {modo}")
            if res.get("top_features"):
                st.markdown("**Por qué (SHAP):**")
                detalle = pd.DataFrame(res["top_features"]).set_index("feature")
                st.bar_chart(detalle["shap"])
    else:
        with col1:
            st.info("Ajusta los valores en la barra lateral y pulsa Predecir.")

    with col2:
        st.subheader("Importancia global de features")
        imp = (pd.Series(art["model"].feature_importances_, index=art["features"])
               .sort_values(ascending=False).head(12))
        st.bar_chart(imp)

    # ---------------- Métricas (punto de operación recomendado) ----------------
    if metrics and "operating_point" in metrics:
        op = metrics["operating_point"]
        st.divider()
        st.subheader(f"Rendimiento del modelo ({metrics['best_model']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{metrics['best']['pr_auc']:.3f}")
        c2.metric("Recall (umbral por coste)", f"{op['recall']*100:.0f}%")
        c3.metric("Umbral", f"{op['threshold']:.2f}")
        c4.metric("Falsas alarmas", op["fp"])
        st.caption(
            f"En el punto de operación recomendado detecta {op['tp']} de "
            f"{op['frauds_total_test']} fraudes del test real y evita ~EUR "
            f"{op['losses_avoided_eur']:,.0f} (coste total ~EUR {op['total_cost_eur']:,.0f})."
        )

    # ---------------- Visualizaciones del EDA ----------------
    st.divider()
    st.subheader("Análisis exploratorio (EDA)")
    imgs = {
        "Distribución de clases": REPORTS / "01_class_distribution.png",
        "Análisis temporal": REPORTS / "03_temporal_analysis.png",
        "Explicabilidad (SHAP)": REPORTS / "12_shap_summary.png",
    }
    for tab, (titulo, ruta) in zip(st.tabs(list(imgs.keys())), imgs.items()):
        with tab:
            if ruta.exists():
                st.image(str(ruta), use_column_width=True)
            else:
                st.warning(f"Falta la imagen {ruta.name}; ejecuta los notebooks para generarla.")


if __name__ == "__main__":
    main()
