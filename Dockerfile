# Imagen para servir la API de fraude. Solo lo necesario para puntuar, nada de notebooks.
FROM python:3.12-slim

WORKDIR /app

# Instalamos dependencias primero para aprovechar la caché de capas de Docker.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# El código y el modelo serializado (best_model.pkl vive dentro de src/).
COPY config.yaml .
COPY src/ ./src/

EXPOSE 8000

# El dataset no se copia: la API solo necesita el modelo ya entrenado.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
