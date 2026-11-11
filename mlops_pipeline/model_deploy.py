
"""
API académica de predicción de créditos.

Carga el pipeline local y recibe uno o varios registros en JSON.
Las transformaciones son las mismas utilizadas en entrenamiento.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ft_engineering import VARIABLES_BASE


ROOT = Path(__file__).resolve().parent.parent


class Credito(BaseModel):
    """Campos requeridos para una predicción."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    tipo_credito: int
    capital_prestado: Annotated[float, Field(gt=0)]
    plazo_meses: Annotated[int, Field(gt=0)]
    edad_cliente: Annotated[int, Field(ge=18, le=100)]
    tipo_laboral: Annotated[str, Field(min_length=1, max_length=100)]
    salario_cliente: Annotated[float, Field(ge=0)]
    total_otros_prestamos: Annotated[float, Field(ge=0)]
    cuota_pactada: Annotated[float, Field(gt=0)]


class SolicitudPrediccion(BaseModel):
    """Limitar la cantidad de registros por solicitud."""

    model_config = ConfigDict(extra="forbid")

    registros: Annotated[
        list[Credito],
        Field(min_length=1, max_length=10000),
    ]


@asynccontextmanager
async def lifespan(app):
    # Cargar una sola vez al iniciar el servidor.
    # No se aceptan modelos enviados por los usuarios.
    app.state.modelo = joblib.load(
        ROOT / "mejor_modelo.joblib"
    )
    yield
    app.state.modelo = None


app = FastAPI(
    title="API de créditos",
    description=(
        "Predicciones académicas por lotes. "
        "No constituye un sistema de aprobación de créditos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    """Comprobar que el servicio y el modelo estén disponibles."""

    disponible = getattr(
        request.app.state, "modelo", None
    ) is not None

    if not disponible:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible.",
        )

    return {"estado": "ok", "modelo_cargado": True}


@app.post("/predict")
def predict(solicitud: SolicitudPrediccion, request: Request):
    """Retornar una predicción por registro, en el mismo orden."""

    modelo = request.app.state.modelo

    datos = pd.DataFrame([
        registro.model_dump()
        for registro in solicitud.registros
    ])[VARIABLES_BASE]

    # La API exige entradas válidas; el pipeline conserva además
    # su tratamiento de datos para la ejecución interna.
    if datos["tipo_laboral"].str.strip().eq("").any():
        raise HTTPException(
            status_code=422,
            detail="tipo_laboral no puede contener solo espacios.",
        )

    predicciones = modelo.predict(datos)

    posicion_clase_0 = list(modelo.classes_).index(0)
    probabilidades = modelo.predict_proba(
        datos
    )[:, posicion_clase_0]

    resultados = [
        {
            "indice": indice,
            "prediccion": int(prediccion),
            "probabilidad_modelo_clase_0": float(probabilidad),
        }
        for indice, (prediccion, probabilidad) in enumerate(
            zip(predicciones, probabilidades)
        )
    ]

    return {
        "cantidad": len(resultados),
        "resultados": resultados,
        "observacion": (
            "Modelo académico con capacidad limitada para la clase 0. "
            "Las probabilidades no están calibradas."
        ),
    }