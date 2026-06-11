# Model Card — Detector de fraude (XGBoost)

Ficha del modelo que sirve la API y el dashboard. La idea es dejar claro, en una página, qué
hace, con qué se entrenó, cómo de bien funciona y —sobre todo— dónde NO hay que confiarse.

## Resumen

- **Tarea:** clasificación binaria — marcar una transacción de tarjeta como fraude (1) o legítima (0).
- **Modelo:** XGBoost (`XGBClassifier`), elegido por PR-AUC frente a Logistic Regression y Random Forest.
- **Versión:** 1.1.0 · **Artefacto:** `src/best_model.pkl` (modelo + scaler + lista de features + umbral).
- **Entrada:** monto, hora y las componentes PCA `V1..V28`. **Salida:** probabilidad de fraude,
  veredicto según el umbral, y las 5 features que más pesaron (SHAP).

## Para qué sirve (y para qué no)

- **Uso previsto:** proyecto de portfolio / demo educativa de un pipeline de detección de fraude
  end-to-end (datos → modelo → API → dashboard).
- **Fuera de alcance:** NO es un sistema de producción. No debe usarse para **bloquear
  transacciones de forma automática** ni tomar decisiones sobre personas reales sin
  revalidación, datos propios y revisión humana. Es un apoyo a la decisión, no la decisión.

## Datos de entrenamiento

- Dataset público de Kaggle (Credit Card Fraud Detection): **284,807** transacciones europeas de
  dos días de 2013, anonimizadas con PCA. Tras quitar **1,081 duplicados** quedan 283,726 filas.
- **Desbalance extremo:** ~0.17 % de fraude.
- Split estratificado 80/20. **SMOTE solo sobre el train**; el test conserva la proporción real.
- Las features `V1..V28` son componentes PCA: protegen la privacidad pero **no son
  interpretables** en términos de negocio (no sabemos qué variable real hay detrás de cada V).

## Cómo de bien funciona

Evaluado sobre el test real (sin SMOTE), 56,746 transacciones:

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.974 |
| PR-AUC | 0.819 |
| PR-AUC (validación cruzada 5-fold) | 0.845 ± 0.034 |
| Baseline trivial (PR-AUC) | 0.002 |

**Punto de operación recomendado** (umbral elegido por coste, no 0.5): umbral **0.15** →
recall **0.83** (79 de 95 fraudes), 65 falsas alarmas, ~€2,803 de coste total estimado
(FN = monto del fraude; FP = ~3 € de revisión manual). Reproducible con `python -m src.pipeline`;
cifras en `reports/metrics.json`.

## Consideraciones éticas y de sesgo

- Los datos están anonimizados; no contienen atributos demográficos directos. Aun así, un modelo
  PCA podría correlacionar con patrones de gasto asociados a ciertos grupos: **no auditable** aquí
  por la propia anonimización.
- **Coste de los errores:** un falso positivo genera fricción al cliente (transacción legítima
  marcada); un falso negativo deja pasar fraude. El umbral por coste hace ese trade-off explícito,
  pero el coste real de cada error lo pone el negocio, no el modelo.
- Cualquier despliegue real debería incluir **revisión humana** de las alertas.

## Límites y mantenimiento

- **Generaliza mal fuera de su dominio:** entrenado con dos días de 2013 en Europa; los patrones
  de fraude cambian (concept/data drift). El rendimiento se degrada con el tiempo.
- **Sin interpretabilidad de negocio** por las features PCA (SHAP explica en términos de `V*`).
- **Umbral atado a un modelo de costes sintético** (€3 por revisión); ajustar a los costes reales.
- **Monitoreo sugerido:** vigilar la distribución de las features de entrada y la tasa de alertas
  en el tiempo; reentrenar cuando se desvíen. El registro `reports/experiments.csv` guarda cada
  corrida para comparar.

## Cómo usarlo

- API: `POST /predict` (ver `src/api.py` y `DOCKER.md`).
- Dashboard: `streamlit run dashboard/app.py` (consume la API; el umbral es ajustable).
- Reentrenar: `python -m src.pipeline`.

---
*Autor: Omar Mora Flores · Última actualización: 2026-06-10*
