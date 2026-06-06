# 🔎 Hallazgos y Aprendizajes — Credit Card Fraud Detection

> Bitácora de **detecciones** (qué nos dijeron los datos y los modelos) y **aprendizajes**
> (qué nos llevamos para el próximo proyecto). Todas las cifras provienen de la ejecución real
> del pipeline y están respaldadas por `reports/metrics.json` y los notebooks `01`–`03`.

**Autor:** Omar Mora Flores · **Última actualización:** 2026-06-06

---

## 🧭 Resumen ejecutivo

Construimos un detector de fraude sobre **284,807 transacciones** con clases extremadamente
desbalanceadas (**0.17 %** de fraude). Tras limpiar duplicados y aplicar feature engineering +
SMOTE (solo en train), **XGBoost** alcanzó **PR-AUC 0.819** y **recall 0.789**, detectando
**75 de 95 fraudes** del conjunto de prueba con apenas **23 falsas alarmas** entre 56,651
transacciones legítimas (~**€10,741** en pérdidas evitadas). El hallazgo metodológico central:
con este nivel de desbalance, *accuracy* y *ROC-AUC* engañan — **PR-AUC y F1 son las métricas
que importan**.

---

## 📊 Detecciones del EDA (Fase 1)

| # | Detección | Evidencia | Decisión que motivó |
|---|---|---|---|
| 1 | **Desbalance extremo:** 0.173 % de fraude (1 por cada ~578 legítimas) | `01_class_distribution.png` | Descartar *accuracy*; balancear el train con SMOTE |
| 2 | **1,081 filas duplicadas** (0.38 %), 0 nulos | `df.duplicated()` en `01_EDA` | Eliminarlas **antes** del split (evitar fuga train↔test) |
| 3 | **`Amount` no separa las clases:** media fraude €122 vs legítima €88, pero medianas y distribuciones solapadas | `02_amount_analysis.png` | Transformar con `log1p` + escalado robusto; no confiar solo en el monto |
| 4 | **Señal temporal:** la *tasa* de fraude sube a **1.7 % entre las 2–4 a.m.** (vs 0.17 % media) | `03_temporal_analysis.png` | Crear features `Hour` e `Is_night` |
| 5 | **Features PCA con más señal:** `V17, V14, V12, V10, V16, V3, V7, V11, V4` | `04_feature_correlation.png` | Anticipar los drivers de la importancia del modelo |

> Nota: las features `V1`–`V28` son componentes PCA anonimizados; su correlación con `Class`
> es el único proxy interpretable de su relevancia.

---

## 🤖 Detecciones del modelado (Fase 3)

Evaluación sobre el **test real** (sin SMOTE), 56,746 transacciones:

| Modelo | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.959 | 0.680 | 0.094 | 0.050 | 0.853 |
| Random Forest | 0.939 | **0.817** | **0.830** | **0.901** | 0.768 |
| **XGBoost** ✅ | **0.974** | **0.819** | 0.777 | 0.765 | 0.789 |

**Lo que detectamos:**

1. **La regresión logística colapsa con el desbalance.** Tiene buen recall (0.85) pero
   precision de **0.05** → genera 17 falsas alarmas por cada acierto. Su ROC-AUC alto (0.959)
   es justamente la trampa que advertimos: oculta una PR-AUC mediocre (0.680).
2. **XGBoost y Random Forest quedan casi empatados** (PR-AUC 0.819 vs 0.817). Se eligió
   XGBoost por su PR-AUC marginalmente superior y mayor ROC-AUC, pero **Random Forest tiene
   mejor precision (0.90) y F1 (0.83)** — sería preferible si el costo de las falsas alarmas
   fuese alto. *La "mejor" elección depende del costo de negocio, no de una sola métrica.*
3. **Matriz de confusión de XGBoost:** VP=75, FN=20, FP=23, VN=56,628 (`08_best_model.png`).
   La curva precision-recall mantiene precision ≈ 1.0 hasta ~0.8 de recall y cae después.
4. **`V14` domina la importancia del modelo**, seguida de `V10` y `V4` — coherente con las
   correlaciones del EDA (`09_feature_importance.png`).

---

## 💰 Impacto de negocio

Reconstruyendo el monto real de cada transacción (inverse-transform del escalado + `expm1`):

- **Pérdidas evitadas:** ~**€10,741** (suma de montos de los 75 fraudes detectados).
- **Pérdidas no capturadas:** ~**€4,025** (los 20 fraudes que se escaparon).
- **Costo operativo:** 23 falsas alarmas a revisar manualmente — manejable para un equipo.

> El recall (cuántos fraudes atrapamos) y la precision (cuánto trabajo en falso generamos)
> son palancas que se ajustan moviendo el *threshold* según el costo real de cada error.

---

## 🎓 Aprendizajes

**Técnicos**
1. **Accuracy y ROC-AUC engañan con clases muy desbalanceadas** — la **PR-AUC** y el **F1**
   describen mucho mejor el rendimiento sobre la clase rara.
2. **SMOTE solo en train, nunca en test.** Aplicarlo en test inventa fraudes que no existieron
   e infla las métricas (data leakage).
3. **El orden importa:** primero `train_test_split`, luego ajustar el `RobustScaler` **solo con
   train**. Escalar antes de partir filtra información del test.
4. **Limpiar antes de partir:** los 1,081 duplicados debían eliminarse antes del split para no
   repartir copias idénticas entre train y test.
5. **Versionar el modelo final** (`src/best_model.pkl`, 900 KB) permite que el dashboard corra
   al clonar, sin re-entrenar ni descargar el dataset.

**De proceso**
6. **Un README con métricas placeholder es deuda, no marketing.** El README inicial afirmaba
   ROC-AUC 0.9996 / "99 % capturado" sin código detrás; lo reemplazamos por cifras reales y
   reproducibles. *Mejor un 79 % honesto que un 99 % inventado.*
7. **Cifras reproducibles > cifras impresionantes:** todo número en el README y aquí se
   regenera ejecutando los notebooks y se guarda en `reports/metrics.json`.
8. **Datasets fuera de git** desde el inicio (`.gitignore`) — un CSV de 150 MB ni siquiera cabe
   en GitHub (límite 100 MB/archivo).

---

## ⚠️ Limitaciones

- Las features PCA son **anónimas** → no hay interpretabilidad de negocio sobre *qué* variable
  real dispara el fraude (mitigable con SHAP, pero seguiría siendo sobre componentes PCA).
- Se evaluó con un **único split**; métricas más estables requerirían validación cruzada
  estratificada o *time-based*.
- El umbral usado es el por defecto (0.5); no se optimizó para un costo de negocio específico.
- El impacto económico asume que cada fraude detectado se previene al 100 %, lo cual es una
  cota superior.

---

## 🚀 Próximos pasos

- [ ] **Tuning de threshold** según matriz de costos (costo de un fraude vs costo de revisar una falsa alarma).
- [ ] **Validación cruzada estratificada** + intervalos de confianza de las métricas.
- [ ] Probar **`scale_pos_weight`** de XGBoost como alternativa a SMOTE (suele dar mejor precision sin sintéticos).
- [ ] **SHAP** para explicar predicciones individuales en el dashboard.
- [ ] Desplegar el dashboard en **Streamlit Cloud**.

---

*Documento vivo — se actualiza conforme evoluciona el proyecto.*
