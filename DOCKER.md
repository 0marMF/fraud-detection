# Servir el modelo con Docker

Guía para empaquetar la API de scoring (`src/api.py`) en un contenedor y levantarla en
cualquier máquina. La imagen es deliberadamente ligera: solo lleva el código de `src/`, el
`config.yaml` y el modelo ya entrenado (`src/best_model.pkl`). No incluye el dataset, los
notebooks ni Jupyter, porque en producción no hacen falta.

## 1. Instalar Docker (una sola vez)

**Windows / macOS:** instala **Docker Desktop**.

- Windows con winget:
  ```powershell
  winget install --id Docker.DockerDesktop -e
  ```
  Requiere permisos de administrador y, normalmente, **WSL2** y un **reinicio**. Tras instalar,
  abre Docker Desktop una vez para que arranque el daemon (el icono de la ballena debe quedar en
  verde). Sin el daemon corriendo, `docker build` no funciona.

- Linux: instala Docker Engine desde el gestor de paquetes de tu distro (`docker.io` o el repo
  oficial) y arranca el servicio `docker`.

Comprueba que está listo:
```bash
docker --version
docker info        # debe responder sin error -> el daemon está activo
```

## 2. Construir la imagen

Desde la raíz del proyecto (donde está el `Dockerfile`):
```bash
docker build -t fraud-api:1.1.0 .
```
La primera vez tarda unos minutos (instala pandas, scikit-learn, xgboost, shap...). El
`.dockerignore` mantiene el contexto pequeño: no copia `data/`, `notebooks/` ni `reports/`.

## 3. Levantar el contenedor

```bash
docker run --rm -p 8000:8000 fraud-api:1.1.0
```
La API queda en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

## 4. Probarla

```bash
# Estado del servicio (modelo y umbral cargados)
curl http://localhost:8000/health

# Puntuar una transacción sospechosa (monto alto, de madrugada, V negativas)
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"amount\": 500, \"hour\": 3, \"v_features\": {\"V14\": -9, \"V10\": -9}}"
```
*(En PowerShell el salto de línea es `^`; en bash usa `\`.)*

Respuesta esperada (aproximada):
```json
{
  "probability": 0.96,
  "is_fraud": true,
  "threshold": 0.15,
  "top_features": [{"feature": "V14", "shap": 5.2}, ...]
}
```

## 5. Conectar el dashboard a la API

El dashboard (`dashboard/app.py`) consume la API si la encuentra. Apúntalo al contenedor con una
variable de entorno y arráncalo:
```bash
# bash
FRAUD_API_URL=http://localhost:8000 streamlit run dashboard/app.py
```
```powershell
# PowerShell
$env:FRAUD_API_URL = "http://localhost:8000"; streamlit run dashboard/app.py
```
Si la API no está disponible, el dashboard cae automáticamente a un modo local (usa la misma
función `src.score.predict`), así que siempre funciona.

## Notas

- El modelo viaja dentro de la imagen (`src/best_model.pkl`), así que el contenedor es
  autosuficiente: no necesita el dataset ni reentrenar.
- Para reentrenar y regenerar el modelo antes de construir la imagen: `python -m src.pipeline`.
- Las dependencias de la imagen están en `requirements-api.txt` (más pequeñas que las de
  desarrollo en `requirements.txt`).
- **Tamaño:** la imagen pesa ~1.9 GB, sobre todo por `shap` (arrastra `numba`/`llvmlite`). Si
  necesitas una imagen mucho más ligera, quita `shap` de `requirements-api.txt` y haz la
  explicación SHAP opcional en `/predict` (la probabilidad y el veredicto no la necesitan).
