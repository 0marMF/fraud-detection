# Roadmap — Credit Card Fraud Detection

**Proyecto de portfolio:** Omar Mora Flores
**Objetivo:** Construir un pipeline completo de detección de fraude con ML, desde EDA hasta un dashboard interactivo desplegable.

---

## Estado general

| Fase | Componente | Estado |
|---|---|---|
| 0 | Setup del entorno | Completado |
| 1 | EDA | Completado |
| 2 | Preprocessing & Feature Engineering | Completado |
| 3 | Modelado & Evaluación | Completado |
| 4 | Dashboard Streamlit | Completado |
| 5 | Cierre de portfolio | Completado |

Leyenda: Pendiente · En progreso · Completado

> **v1.0.0 completo.** Más abajo están el **Backlog de mejoras (v1.1)** y el **Track de Ciencia de Datos** — el plan
> de esta segunda iteración.

---

## Backlog de mejoras — v1.1.0 *(en progreso: CP1-CP2 hechos)*

> Derivado de la revisión de calidad (2026-06-06). Corrige focos que la v1.0.0 dejó a medias.

- [x] **3.7 Selección de umbral por matriz de costes** — umbral óptimo por coste en euros = 0.15 (recall 0.83, ~€2,803 de coste total vs ~€4,094 a 0.5). Curva en `reports/11_threshold_cost.png`.
- [x] **3.8 Explicabilidad por predicción (SHAP)** — `TreeExplainer` + summary (`12_shap_summary.png`) y waterfall por transacción (`13_shap_waterfall.png`); lógica reutilizable en `src/explain.py` y notebook `04_explainability.ipynb`. La API y el dashboard ya muestran las features SHAP por predicción (CP5).
- [x] **Validación cruzada estratificada (`StratifiedKFold`)** — PR-AUC 0.845 ± 0.034 (SMOTE dentro de cada fold).
- [x] **Baseline `DummyClassifier`** — PR-AUC ≈ 0.002 · *pendiente: comparar `scale_pos_weight` como alternativa a SMOTE*.
- [x] Liderar la narrativa de métricas con **PR-AUC** (no ROC-AUC) en README y notebooks.

> Infra del Track DS ya en marcha: `config.yaml`, paquete `src/` modular, `python -m src.pipeline`
> y registro de experimentos (`reports/experiments.csv`), SHAP (`src/explain.py`), tests
> (`pytest`, 12 verdes) y API + Docker (`src/api.py`, `Dockerfile`). Pendiente:
> CI (CP6), model card (CP7), release v1.1.0 (CP8).

---

## Fase 0 — Setup del entorno

**Meta:** El proyecto debe poder reproducirse desde cero con un solo comando.

### Tareas

- [x] Completar `requirements.txt` con todas las dependencias y versiones fijas
- [x] Completar `.gitignore` (excluir `data/`, `*.csv`, `*.pkl`, `*.zip`, `__pycache__/`, `.ipynb_checkpoints/`)
- [x] Actualizar rutas de notebooks en `README.md` de `.py` a `.ipynb` · *pendiente: reemplazar las métricas placeholder por las reales en la Fase 3/5*
- [x] Verificar que el dataset `data/creditcard.csv` carga correctamente con Pandas (284,807 filas, 0 nulos, 1,081 duplicados)

### Entregables
- `requirements.txt` funcional
- `.gitignore` completo
- Proyecto ejecutable end-to-end

---

## Fase 1 — EDA (Análisis Exploratorio)

**Archivo:** `notebooks/01_EDA.ipynb`
**Meta:** Entender la distribución de los datos, el desbalance de clases y los patrones que diferencian transacciones fraudulentas de legítimas.

### Secciones del notebook

#### 1.1 Carga y descripción del dataset
- [x] Cargar `creditcard.csv` con Pandas
- [x] Mostrar shape, dtypes, primeras filas, y estadísticas descriptivas (`df.describe()`)
- [x] Verificar valores nulos y duplicados → **0 nulos, 1,081 duplicados detectados**

#### 1.2 Distribución de clases
- [x] Contar y calcular porcentaje de fraudes vs. legítimas → **0.173% fraude**
- [x] Gráfico de barras con conteo y porcentaje por clase
- [x] Guardar → `reports/01_class_distribution.png`

#### 1.3 Análisis del monto (`Amount`)
- [x] Boxplot de Amount por clase (fraude vs. legítima)
- [x] Histograma de Amount con escala log para mejor visualización
- [x] Estadísticas descriptivas de Amount separadas por clase
- [x] Guardar → `reports/02_amount_analysis.png`

#### 1.4 Análisis temporal
- [x] Convertir `Time` (segundos) a hora del día: `hora = (df['Time'] / 3600) % 24`
- [x] Distribución de transacciones por hora separada por clase
- [x] Identificar patrones horarios de fraude → **pico de tasa de fraude a las 2–4 a.m.**
- [x] Guardar → `reports/03_temporal_analysis.png`

#### 1.5 Correlación de features con la variable objetivo
- [x] Calcular correlación de Pearson de todas las features con `Class`
- [x] Heatmap de correlaciones (solo features con |corr| > 0.1 para legibilidad)
- [x] Identificar las 10 features más correlacionadas con fraude → **V17, V14, V12, V10, V16…**
- [x] Guardar → `reports/04_feature_correlation.png`

#### 1.6 Conclusiones del EDA
- [x] Celda Markdown con hallazgos clave que justifican las decisiones de preprocessing

### Entregables
- `notebooks/01_EDA.ipynb` ejecutable de inicio a fin sin errores
- 4 imágenes en `reports/`

---

## Fase 2 — Preprocessing & Feature Engineering

**Archivo:** `notebooks/02_preprocessing.ipynb`
**Meta:** Limpiar, transformar y enriquecer el dataset para maximizar la capacidad predictiva del modelo.

### Secciones del notebook

#### 2.1 Carga del dataset limpio

- [x] Cargar `creditcard.csv`
- [x] Confirmar 0 nulos y **eliminar los 1,081 duplicados** detectados en el EDA

#### 2.2 Feature Engineering

- [x] Crear `Hour`: hora del día desde `Time`
  ```python
  df['Hour'] = (df['Time'] / 3600) % 24
  ```
- [x] Crear `Amount_log`: logaritmo del monto para reducir skewness
  ```python
  df['Amount_log'] = np.log1p(df['Amount'])
  ```
- [x] Crear `Is_night`: bandera binaria para transacciones entre 22:00–06:00
  ```python
  df['Is_night'] = df['Hour'].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)
  ```
- [x] Eliminar columnas originales reemplazadas: `Time`, `Amount`

#### 2.3 Escalado de features

- [x] Aplicar `RobustScaler` a `Amount_log` y `Hour` (robusto a outliers)
- [x] Las features V1–V28 ya vienen escaladas por PCA — no re-escalar

#### 2.4 Split train/test

- [x] Separar features `X` y target `y`
- [x] Split estratificado 80/20 con `stratify=y` para preservar proporción de fraudes
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  ```

#### 2.5 Balanceo de clases — SMOTE

- [x] Aplicar SMOTE **solo sobre el conjunto de entrenamiento**
- [x] Verificar distribución antes y después del SMOTE
- [x] Documentar explícitamente por qué NO se aplica en el test set (data leakage)

#### 2.6 Guardar splits procesados

- [x] Serializar con `pickle` en `data/splits.pkl`:
  ```python
  splits = {'X_train': X_train_res, 'X_test': X_test,
            'y_train': y_train_res, 'y_test': y_test}
  ```

### Entregables
- `notebooks/02_preprocessing.ipynb` ejecutable end-to-end
- `data/splits.pkl` con los 4 arrays listos para modelado

---

## Fase 3 — Modelado & Evaluación

**Archivo:** `notebooks/03_modeling.ipynb`
**Meta:** Entrenar, comparar y seleccionar el mejor modelo. Serializar el modelo final.

### Secciones del notebook

#### 3.1 Carga de splits

- [x] Cargar `data/splits.pkl`
- [x] Confirmar shapes y distribución de clases en train/test

#### 3.2 Entrenamiento de modelos baseline

- [x] **Logistic Regression** — baseline simple
- [x] **Random Forest** — ensemble clásico
- [x] **XGBoost** — modelo principal esperado

Para cada modelo calcular:
- [x] ROC-AUC
- [x] F1-Score (clase fraude)
- [x] Precision y Recall
- [x] Matriz de confusión

#### 3.3 Visualización comparativa

- [x] Tabla comparativa de métricas por modelo
- [x] Curvas ROC superpuestas para los 3 modelos
- [x] Guardar → `reports/06_model_comparison.png`
- [x] Guardar → `reports/07_roc_curves.png`

#### 3.4 Análisis del mejor modelo (XGBoost)

- [x] Matriz de confusión detallada con anotaciones
- [x] Curva Precision-Recall
- [x] Feature importance (top 15 features)
- [x] Guardar → `reports/08_best_model.png`
- [x] Guardar → `reports/09_feature_importance.png`

#### 3.5 Serialización del modelo

- [x] Guardar modelo final en `src/best_model.pkl`
  ```python
  import pickle
  with open('src/best_model.pkl', 'wb') as f:
      pickle.dump(best_model, f)
  ```

#### 3.6 Impacto de negocio

- [x] Calcular fraudes detectados / total fraudes en test set
- [x] Estimar monto de pérdidas evitadas vs. costo de falsas alarmas
- [x] Presentar resumen ejecutivo en Markdown

### Entregables
- `notebooks/03_modeling.ipynb` ejecutable end-to-end
- `src/best_model.pkl` serializado
- 4 imágenes en `reports/`

---

## Fase 4 — Dashboard Streamlit

**Archivo:** `dashboard/app.py`
**Meta:** Interfaz interactiva que permita ingresar datos de transacción y obtener predicción de fraude en tiempo real.

### Secciones del dashboard

#### 4.1 Estructura base

- [x] Crear carpeta `dashboard/` si no existe
- [x] Crear `dashboard/app.py` con estructura Streamlit básica

#### 4.2 Sidebar — inputs del usuario

- [x] Sliders / inputs para las features más importantes (Amount, Hour, top V-features)
- [x] Botón "Predecir"

#### 4.3 Panel principal — resultados

- [x] Indicador visual: FRAUDE / LEGÍTIMA con color (rojo/verde)
- [x] Probabilidad de fraude con barra de progreso
- [x] Gráfico de feature importance del modelo cargado

#### 4.4 Sección de métricas del modelo

- [x] Mostrar métricas clave del modelo (ROC-AUC, F1, etc.)
- [x] Mostrar las visualizaciones del EDA como referencia

#### 4.5 Carga del modelo

- [x] Cargar `src/best_model.pkl` con `@st.cache_resource`
- [x] Manejar errores si el modelo no existe aún

### Entregables
- `dashboard/app.py` funcional ejecutable con `streamlit run dashboard/app.py`

---

## Fase 5 — Cierre de portfolio

**Meta:** El proyecto debe verse profesional y ser 100% reproducible por cualquier persona que lo clone.

### Tareas

- [x] Ejecutar todos los notebooks de inicio a fin sin errores (01 → 02 → 03, 0 errores)
- [x] Actualizar `README.md` con screenshots reales del dashboard → `reports/10_dashboard.png`
- [x] Revisar que `requirements.txt` incluye todas las dependencias usadas
- [x] Crear tag de versión en git: `v1.0.0` → [release publicado](https://github.com/0marMF/fraud-detection/releases/tag/v1.0.0)
- [ ] (Opcional) Desplegar dashboard en Streamlit Cloud

### Checklist de calidad del portfolio

- [x] Los notebooks tienen narrativa en Markdown entre las celdas de código
- [x] Los gráficos tienen títulos, etiquetas de ejes y leyendas en español
- [x] No hay rutas absolutas hardcodeadas — solo rutas relativas (raíz auto-localizada)
- [x] El modelo reproducible con `random_state=42` en todos los pasos
- [x] El README tiene las métricas finales reales (no las del placeholder)

---

## De análisis a proyecto de Ciencia de Datos (Track DS — planificado)

Hasta aquí esto es un análisis sólido. Pero un científico de datos no entrega un notebook bonito
y se va: entrega algo que otra persona puede clonar, correr, probar, desplegar y mantener sin
adivinar nada. Esta es la parte que convierte el proyecto de "análisis" en "proyecto de DS".

**Sacar el código del notebook → `src/`**
- [ ] `src/data.py` (carga y limpieza), `src/features.py` (Hour, Amount_log, Is_night, scaler),
      `src/model.py` (entrenamiento + serialización), `src/evaluate.py` (métricas y umbral).
- [ ] Los notebooks pasan a contar la historia y llamar a esas funciones — se acaba el copy-paste
      entre `02` y `03`.

**Reproducibilidad de verdad**
- [ ] `config.yaml` con rutas, semilla, hiperparámetros y umbral. Nada hardcodeado.
- [ ] Un solo comando que corre todo: `python -m src.pipeline` (o un `Makefile`).

**Tests con pytest** (esto es lo que más separa a un DS de un analista) — hecho (10 verdes, datos sintéticos)
- [x] Validación de datos al cargar: columnas esperadas, rangos, `Class ∈ {0,1}`.
- [x] Test anti-leakage: SMOTE solo en train; el scaler se ajusta solo con train.
- [x] Test de contrato del modelo: `predict_proba` devuelve valores en [0,1].

**Seguimiento de experimentos**
- [ ] MLflow (o, si se quiere ligero, un `experiments.csv`) con params + métricas + artefacto por
      corrida, para poder comparar "qué probé y qué salió".

**Servir el modelo** — hecho (CP5)
- [x] API con FastAPI (`src/api.py`): `POST /predict` devuelve probabilidad, veredicto (según el
      umbral por coste) y las features que más pesaron (SHAP). Lógica compartida en `src/score.py`.
- [x] `Dockerfile` + `requirements-api.txt` para levantar la API de forma reproducible (imagen
      lean: sin notebooks ni jupyter).
- [x] El dashboard consume la API (con fallback local a `src.score`) en vez de cargar el `.pkl`.

**CI**
- [ ] GitHub Actions que corre `pytest` en cada push (y si se puede, ejecuta los notebooks para que
      no se rompan en silencio).

**Documentación de modelo**
- [ ] Model card: para qué sirve y para qué NO, con qué datos se entrenó, límites y sesgos
      conocidos (las `V` son PCA anónimas → poca interpretabilidad de negocio).
- [ ] Nota de monitoreo: cómo detectaríamos *data drift* (los patrones de fraude cambian; un
      modelo viejo se degrada solo).

> **Estilo — vale para TODO lo que escribamos aquí:** el código, los comentarios y la
> documentación deben leerse como escritos por una persona, no por una máquina. Comentarios que
> explican el *porqué* y no el *qué*, nombres honestos, y algún apunte real ("ojo con esto",
> "esto me costó pillarlo"). Cero relleno robótico, cero párrafos de plantilla.

---

## Orden de desarrollo recomendado

```
Fase 0 → Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5
  Setup    EDA    Preproc   Model    Dashboard  Cierre
```

Cada fase produce los archivos que necesita la siguiente.
No saltar fases — el `splits.pkl` de la Fase 2 es requerido por la Fase 3.

---

## Notas técnicas

- `random_state=42` en todo el pipeline para reproducibilidad
- SMOTE **solo en train**, nunca en test (evitar data leakage)
- Usar rutas relativas desde la raíz del proyecto
- Dataset NO se sube a git (está en `.gitignore`) — agregar instrucción de descarga al README
