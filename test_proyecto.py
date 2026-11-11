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

@pytest.mark.parametrize("ruta_absoluta", [False, True])
def test_monitoreo_rechaza_ruta_externa(
    tmp_path, monkeypatch, credito, ruta_absoluta
):
    """Rechazar rutas externas aunque el CSV exista."""
    carpeta_proyecto = tmp_path / "proyecto"
    carpeta_proyecto.mkdir()
    monkeypatch.setattr(monitoreo, "ROOT", carpeta_proyecto)

    referencia = pd.DataFrame([credito] * 30)
    archivo_externo = tmp_path / "externo.csv"
    referencia.to_csv(archivo_externo, index=False)

    ruta = archivo_externo if ruta_absoluta else "../externo.csv"

    with pytest.raises(ValueError, match="dentro de la carpeta"):
        monitoreo.procesar_y_guardar_lote(ruta, referencia)

    # El rechazo debe ocurrir antes de crear la base de datos.
    assert not (carpeta_proyecto / "monitoreo.sqlite").exists()


def test_monitoreo_rechaza_archivo_no_csv(
    tmp_path, monkeypatch, credito
):
    """Evitar procesar archivos con otra extensión."""
    monkeypatch.setattr(monitoreo, "ROOT", tmp_path)
    archivo = tmp_path / "datos.txt"
    archivo.write_text("contenido de prueba", encoding="utf-8")
    referencia = pd.DataFrame([credito] * 30)

    with pytest.raises(ValueError, match="archivo CSV"):
        monitoreo.procesar_y_guardar_lote(archivo, referencia)

    assert not (tmp_path / "monitoreo.sqlite").exists()

def test_health_sin_modelo_devuelve_503(client, monkeypatch):
    """Informar cuando el modelo no esta disponible."""
    monkeypatch.setattr(app.state, "modelo", None)

    respuesta = client.get("/health")

    assert respuesta.status_code == 503
    assert respuesta.json()["detail"] == "Modelo no disponible."

def test_lifespan_libera_modelo_al_cerrar():
    """Liberar el modelo al terminar el contexto del servidor."""
    with TestClient(app) as cliente:
        assert cliente.get("/health").status_code == 200
        assert app.state.modelo is not None

    assert app.state.modelo is None

def test_lifespan_libera_modelo_ante_error():
    """Una excepcion no debe impedir la limpieza."""
    with pytest.raises(RuntimeError, match="Error simulado"):
        with TestClient(app):
            assert app.state.modelo is not None
            raise RuntimeError("Error simulado")

    assert app.state.modelo is None

def test_metricas_clase_minoritaria():
    """Verificar las metricas con resultados calculables a mano."""
    import model_training_evaluation as entrenamiento

    metricas = entrenamiento.summarize_classification(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    )

    assert metricas["accuracy"] == pytest.approx(0.75)
    assert metricas["balanced_accuracy"] == pytest.approx(0.75)
    assert metricas["precision_clase_0"] == pytest.approx(1.0)
    assert metricas["recall_clase_0"] == pytest.approx(0.5)
    assert metricas["f1_clase_0"] == pytest.approx(2 / 3)


def test_comparacion_y_guardado_temporal(tmp_path, monkeypatch, credito):
    """Comparar modelos y verificar sus archivos sin tocar los reales."""
    import joblib
    import model_training_evaluation as entrenamiento
    from sklearn.dummy import DummyClassifier

    # Datos sinteticos exclusivos para probar el funcionamiento.
    registros = [
        {
            **credito,
            "capital_prestado": 1000000 + i * 20000,
            "edad_cliente": 25 + i % 40,
            "salario_cliente": 2000000 + i * 10000,
            "tipo_laboral": "Empleado" if i % 2 else "Independiente",
        }
        for i in range(60)
    ]
    entradas = pd.DataFrame(registros)
    etiquetas = pd.Series([0 if i % 4 == 0 else 1 for i in range(60)])

    resumen = entrenamiento.comparar_modelos(entradas, etiquetas)

    assert set(resumen["modelo"]) == {
        "Base_mayoritaria",
        "Regresion_logistica",
        "Arbol_decision",
        "Random_forest",
    }
    assert len(resumen) == 4
    assert resumen["f1_clase_0_cv"].is_monotonic_decreasing

    columnas_metricas = [
        "f1_clase_0_cv",
        "precision_clase_0_cv",
        "recall_clase_0_cv",
        "balanced_accuracy_cv",
        "accuracy_cv",
    ]

    for columna in columnas_metricas:
        assert resumen[columna].between(0, 1).all()

    assert np.isfinite(
        resumen.select_dtypes(include="number").to_numpy()
    ).all()

    base = resumen.set_index("modelo").loc["Base_mayoritaria"]
    assert base["f1_clase_0_cv"] == pytest.approx(0)
    assert base["accuracy_cv"] == pytest.approx(0.75)

    # Usar un modelo sencillo para comprobar la persistencia.
    modelo = entrenamiento.build_model(
        DummyClassifier(strategy="most_frequent", random_state=42)
    )
    modelo.fit(entradas, etiquetas)
    predicciones = modelo.predict(entradas)
    metricas = entrenamiento.summarize_classification(
        etiquetas, predicciones
    )

    # guardar_resultados calcula su destino a partir de __file__.
    # Redirigirlo a una carpeta temporal, sin sobrescribir el proyecto.
    monkeypatch.setattr(
        entrenamiento,
        "__file__",
        str(tmp_path / "mlops_pipeline" / "model_training_evaluation.py"),
    )

    entrenamiento.guardar_resultados(
        resumen, modelo, metricas, etiquetas, predicciones
    )

    tabla = pd.read_csv(tmp_path / "comparacion_modelos.csv")
    assert tabla["modelo"].tolist() == resumen["modelo"].tolist()

    guardadas = json.loads(
        (tmp_path / "metricas_prueba.json").read_text(encoding="utf-8")
    )
    assert guardadas == pytest.approx(metricas)

    recuperado = joblib.load(tmp_path / "mejor_modelo.joblib")
    np.testing.assert_array_equal(
        recuperado.predict(entradas), predicciones
    )

    for nombre in [
        "comparacion_f1.png",
        "comparacion_sobreajuste.png",
        "matriz_confusion_prueba.png",
    ]:
        imagen = tmp_path / nombre
        assert imagen.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert imagen.stat().st_size > 1000

@pytest.mark.parametrize(
    "estado_historial",
    ["inexistente", "vacio", "con_datos", "danado"],
)
def test_streamlit_demostracion_e_historial(
    tmp_path, monkeypatch, estado_historial
):
    """Probar predicciones, graficas e historial sin tocar la base real."""
    from streamlit.testing.v1 import AppTest
    from ft_engineering import cargar_y_separar_datos

    ruta_bd = tmp_path / "historial_prueba.sqlite"
    monkeypatch.setenv("CREDITOS_HISTORIAL_DB", str(ruta_bd))

    if estado_historial == "danado":
        ruta_bd.write_bytes(b"Esto no es una base SQLite")
    elif estado_historial != "inexistente":
        with closing(sqlite3.connect(ruta_bd)) as conexion, conexion:
            monitoreo.crear_tablas(conexion)

            if estado_historial == "con_datos":
                conexion.execute(
                    "INSERT INTO lotes VALUES (?, ?, ?, ?, ?)",
                    ("lote-prueba", "firma-prueba", "2026-11-01T12:00:00Z", 1, 0),
                )
                conexion.execute(
                    "INSERT INTO drift VALUES (?, ?, ?, ?)",
                    (
                        "lote-prueba",
                        "capital_prestado",
                        "SIN_ALERTA",
                        json.dumps({
                            "variable": "capital_prestado",
                            "estado": "SIN_ALERTA",
                            "psi": 0.0,
                        }),
                    ),
                )
                conexion.execute(
                    "INSERT INTO predicciones VALUES (?, ?, ?)",
                    (
                        "lote-prueba",
                        0,
                        json.dumps({
                            "capital_prestado": 2000000,
                            "prediccion": 1,
                            "probabilidad_clase_0": 0.2,
                        }),
                    ),
                )

    simulacion = AppTest.from_file(
        str(ROOT / "app.py"), default_timeout=60
    ).run()

    assert not simulacion.exception
    assert len(simulacion.title) == 1
    assert len(simulacion.tabs) == 3

    _, prueba, _, _ = cargar_y_separar_datos()
    assert int(simulacion.metric[0].value) == len(prueba)

    # Verificar las predicciones mostradas y sus probabilidades.
    salida = simulacion.dataframe[0].value
    assert len(salida) == min(100, len(prueba))
    assert salida["prediccion"].isin([0, 1]).all()
    assert salida["probabilidad_modelo_clase_0"].between(0, 1).all()

    mensajes = [elemento.value for elemento in simulacion.info]

    if estado_historial == "inexistente":
        assert any("no existe historial" in mensaje for mensaje in mensajes)
        assert not ruta_bd.exists()
    elif estado_historial == "vacio":
        assert any("No hay lotes registrados" in mensaje for mensaje in mensajes)
    elif estado_historial == "danado":
        assert any(
            "No se pudo consultar el historial" in elemento.value
            for elemento in simulacion.error
        )
    else:
        tablas = [elemento.value for elemento in simulacion.dataframe]
        tabla_lotes = next(tabla for tabla in tablas if "id" in tabla.columns)
        assert tabla_lotes["id"].tolist() == ["lote-prueba"]

        tabla_predicciones = next(
            tabla for tabla in tablas
            if "probabilidad_clase_0" in tabla.columns
        )
        assert tabla_predicciones.iloc[0]["prediccion"] == 1

        with closing(sqlite3.connect(ruta_bd)) as conexion:
            assert conexion.execute(
                "SELECT COUNT(*) FROM lotes"
            ).fetchone()[0] == 1

    # Cambiar de grafica categorica a numerica.
    selector = next(
        elemento for elemento in simulacion.selectbox
        if elemento.label == "Variable para comparar"
    )
    selector.select("capital_prestado").run()
    assert not simulacion.exception

    # Verificar que actualizar el historial no provoque excepciones.
    boton = next(
        elemento for elemento in simulacion.button
        if elemento.label == "Actualizar historial"
    )
    boton.click().run()
    assert not simulacion.exception

@pytest.mark.parametrize(
    "valores,esperado",
    [
        ([0, 1], "DATOS_INSUFICIENTES"),
        ([0, 1, 2], "TENDENCIA_CRECENTE"),
        ([2, 1, 0], "TENDENCIA_DECRECIENTE"),
        ([0, 1, 4], "CAMBIO_ABRUPTO"),
        ([4, 3, 0], "CAMBIO_ABRUPTO"),
        ([1, 1, 1], "SIN_TENDENCIA_DETECTADA"),
    ],
)
def test_evolucion_del_historial(valores, esperado):
    """Verificar tendencias, cambios abruptos y datos insuficientes."""
    lotes = pd.DataFrame({
        "fecha_utc": pd.date_range(
            "2026-01-01", periods=len(valores), tz="UTC"
        ),
        "variables_alerta": valores,
    })

    # La funcion debe ordenar las ejecuciones por su fecha.
    resultado = monitoreo.analizar_evolucion(lotes.iloc[::-1])
    assert resultado == esperado
