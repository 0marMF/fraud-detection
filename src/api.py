"""API de scoring de fraude (FastAPI).

Levantar en local:  uvicorn src.api:app --reload
Documentación interactiva en  http://localhost:8000/docs

Expone el modelo como un servicio: le mandas una transacción y te devuelve la probabilidad, el
veredicto (según el umbral por coste que guardamos) y por qué. El modelo se carga una sola vez.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .score import load_artifact, predict

app = FastAPI(
    title="Fraud Detection API",
    version="1.1.0",
    description="Scoring de fraude en transacciones con XGBoost + explicación SHAP.",
)

_artefacto: dict | None = None


def _art() -> dict:
    # Carga perezosa: el modelo se lee del disco la primera vez y se reutiliza.
    global _artefacto
    if _artefacto is None:
        _artefacto = load_artifact()
    return _artefacto


class Transaction(BaseModel):
    amount: float = Field(..., ge=0, description="Monto de la transacción en euros")
    hour: float = Field(..., ge=0, le=24, description="Hora del día (0-24)")
    v_features: dict[str, float] = Field(
        default_factory=dict,
        description="Componentes PCA V1..V28. Las que no se envíen se asumen 0.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"amount": 149.99, "hour": 3, "v_features": {"V14": -8.0, "V10": -6.0}}
        }
    }


@app.get("/health")
def health() -> dict:
    art = _art()
    return {"status": "ok", "model": art["model_name"], "threshold": art["threshold"]}


@app.post("/predict")
def predict_endpoint(tx: Transaction) -> dict:
    """Devuelve probabilidad de fraude, veredicto y las features que más pesaron."""
    return predict(_art(), tx.amount, tx.hour, tx.v_features)
