"""API de segmentación de clientes. Carga el K-Means entrenado y lo expone."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"


def project_root() -> Path:
    override = os.getenv("APP_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], here.parents[2], Path("/app")):
        if (candidate / "models" / "kmeans_mall.joblib").exists():
            return candidate
    return here.parents[2]


ROOT = project_root()
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "kmeans_mall.joblib"))
ANALYSIS_PATH = Path(os.getenv("ANALYSIS_PATH", ROOT / "artifacts" / "analysis.json"))

app = FastAPI(
    title="Segmentación de clientes — Mall",
    description="K-Means sobre el Mall Customers Dataset: exploración, codo, silueta y clasificación.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClienteNuevo(BaseModel):
    gender: str = Field(pattern="^(Female|Male)$")
    age: int = Field(ge=15, le=100)
    annual_income: float = Field(ge=0, le=300)
    spending_score: float = Field(ge=1, le=100)


@lru_cache(maxsize=1)
def artifacts() -> tuple[dict, dict]:
    if not MODEL_PATH.exists() or not ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            "No está el modelo o el análisis. Ejecuta: python ml/train.py"
        )
    model = joblib.load(MODEL_PATH)
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    return model, analysis


@app.get("/api/health")
def health() -> dict:
    ready = MODEL_PATH.exists() and ANALYSIS_PATH.exists()
    return {"status": "ok" if ready else "sin_modelo", "modelo": str(MODEL_PATH.name)}


@app.get("/api/analysis")
def analysis() -> dict:
    try:
        _, payload = artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    public = dict(payload)
    public.pop("customers", None)
    return public


@app.get("/api/customers")
def customers() -> dict:
    try:
        _, payload = artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"customers": payload["customers"], "clusters": payload["clusters"]}


@app.post("/api/predict")
def predict(cliente: ClienteNuevo) -> dict:
    try:
        model, payload = artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pipeline = model["pipeline"]
    features = model["features"]
    frame = pd.DataFrame(
        [{INCOME: cliente.annual_income, SPENDING: cliente.spending_score}],
        columns=features,
    )
    cluster_id = int(pipeline.predict(frame)[0])
    scaled = pipeline.named_steps["scaler"].transform(frame)
    center = pipeline.named_steps["kmeans"].cluster_centers_[cluster_id]
    distance = float(np.linalg.norm(scaled[0] - center))
    profile = next(item for item in payload["clusters"] if item["id"] == cluster_id)
    return {
        "cluster": cluster_id,
        "distancia_al_centroide": round(distance, 4),
        "perfil": profile,
        "entrada": cliente.model_dump(),
        "nota": (
            "La clasificación usa ingreso anual y puntaje de gasto, ya escalados "
            "con el StandardScaler del entrenamiento. Edad y género se guardan como "
            "contexto del cliente y no entran en la distancia."
        ),
    }
