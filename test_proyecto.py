"""
Pruebas del proyecto de créditos.

Se verifican entradas inválidas, predicciones por lote,
transformaciones, alertas de drift y persistencia sin duplicados.
"""

import json
import sqlite3
import sys
from pathlib import Path
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mlops_pipeline"))

import model_monitoring as monitoreo
from ft_engineering import crear_pipeline_preparacion
from model_deploy import app


@pytest.fixture
def credito():
    """Registro de ejemplo válido para las pruebas."""
    return {
        "tipo_credito": 4,
        "capital_prestado": 2000000,
        "plazo_meses": 12,
        "edad_cliente": 35,
        "tipo_laboral": "Empleado",
        "salario_cliente": 3000000,
        "total_otros_prestamos": 500000,
        "cuota_pactada": 200000,
    }


@pytest.fixture
def client():
    # El contexto ejecuta la carga y cierre del modelo.
    with TestClient(app) as cliente:
        yield cliente


def test_health(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["modelo_cargado"] is True


def test_prediccion_por_lote(client, credito):
    segundo = {
        **credito,
        "capital_prestado": 1500000,
        "edad_cliente": 42,
        "tipo_laboral": "Independiente",
    }

    registros = [credito, segundo]
    respuesta = client.post(
        "/predict",
        json={"registros": registros},
    )

    assert respuesta.status_code == 200
    resultados = respuesta.json()["resultados"]
    assert respuesta.json()["cantidad"] == 2

    # Verificar que la API preserve el orden y coincida con el modelo.
    esperadas = app.state.modelo.predict(pd.DataFrame(registros))

    for indice, resultado in enumerate(resultados):
        assert resultado["indice"] == indice
        assert resultado["prediccion"] == int(esperadas[indice])
        assert 0 <= resultado["probabilidad_modelo_clase_0"] <= 1


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("capital_prestado", -1),
        ("plazo_meses", 0),
        ("edad_cliente", 150),
        ("salario_cliente", -100),
        ("tipo_laboral", "   "),
    ],
)
def test_api_rechaza_valores_invalidos(client, credito, campo, valor):
    registro = {**credito, campo: valor}

    respuesta = client.post(
        "/predict",
        json={"registros": [registro]},
    )

    assert respuesta.status_code == 422


def test_api_rechaza_lote_vacio(client):
    respuesta = client.post("/predict", json={"registros": []})
    assert respuesta.status_code == 422


def test_api_rechaza_campo_faltante(client, credito):
    registro = credito.copy()
    del registro["cuota_pactada"]

    respuesta = client.post(
        "/predict",
        json={"registros": [registro]},
    )

    assert respuesta.status_code == 422


def test_pipeline_nulos_y_categoria_nueva(credito):
    entrenamiento = pd.DataFrame([
        credito,
        {**credito, "edad_cliente": 45, "salario_cliente": 4000000},
    ])

    nuevo = pd.DataFrame([{
        **credito,
        "edad_cliente": 123,
        "salario_cliente": 0,
        "tipo_laboral": "Categoria_nueva",
    }])

    original = nuevo.copy(deep=True)
    pipeline = crear_pipeline_preparacion()
    pipeline.fit(entrenamiento)

    resultado = pipeline.transform(nuevo)

    assert resultado.shape[0] == 1
    assert np.isfinite(resultado).all()
    pd.testing.assert_frame_equal(nuevo, original)


def test_psi_referencia_constante():
    referencia = np.ones(100)

    assert monitoreo.calcular_psi(referencia, referencia) == pytest.approx(0)
    assert monitoreo.calcular_psi(
        referencia, np.full(100, 10)
    ) > 0.20


def test_drift_detecta_cambio(credito):
    referencia = pd.DataFrame([credito] * 100)
    actual = referencia.copy()
    actual["capital_prestado"] *= 10

    reporte = monitoreo.analizar_drift(
        referencia, actual
    ).set_index("variable")

    assert reporte.loc["capital_prestado", "estado"] == "REVISAR"
    assert reporte.loc["tipo_laboral", "estado"] == "SIN_ALERTA"


def test_drift_rechaza_columnas_faltantes(credito):
    referencia = pd.DataFrame([credito] * 30)
    actual = referencia.drop(columns=["edad_cliente"])

    with pytest.raises(ValueError, match="Faltan columnas"):
        monitoreo.analizar_drift(referencia, actual)


def test_drift_reporta_datos_insuficientes(credito):
    referencia = pd.DataFrame([credito] * 30)
    actual = pd.DataFrame([credito])

    reporte = monitoreo.analizar_drift(referencia, actual)
    assert reporte["estado"].eq("DATOS_INSUFICIENTES").all()


def test_guardado_lote_sin_duplicados(tmp_path, monkeypatch, credito):
    # Usar una carpeta temporal para no modificar el historial real.
    monkeypatch.setattr(monitoreo, "ROOT", tmp_path)

    (tmp_path / "mejor_modelo.joblib").write_bytes(
        (ROOT / "mejor_modelo.joblib").read_bytes()
    )

    referencia = pd.DataFrame([credito] * 30)
    ruta_lote = tmp_path / "lote.csv"
    referencia.to_csv(ruta_lote, index=False)

    monitoreo.procesar_y_guardar_lote(ruta_lote, referencia)
    monitoreo.procesar_y_guardar_lote(ruta_lote, referencia)

    with closing(
        sqlite3.connect(tmp_path / "monitoreo.sqlite")
    ) as conexion:
        cantidad_lotes = conexion.execute(
            "SELECT COUNT(*) FROM lotes"
        ).fetchone()[0]

        cantidad_predicciones = conexion.execute(
            "SELECT COUNT(*) FROM predicciones"
        ).fetchone()[0]

        cantidad_metricas = conexion.execute(
            "SELECT COUNT(*) FROM drift"
        ).fetchone()[0]

        registro = json.loads(conexion.execute(
            "SELECT datos_json FROM predicciones LIMIT 1"
        ).fetchone()[0])

    assert cantidad_lotes == 1
    assert cantidad_predicciones == 30
    assert cantidad_metricas == 8
    assert registro["prediccion"] in [0, 1]
    assert 0 <= registro["probabilidad_clase_0"] <= 1
