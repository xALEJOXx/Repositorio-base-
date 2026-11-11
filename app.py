"""
Aplicación académica de créditos.

Permite cargar un CSV, generar predicciones y comparar su población
con los datos de entrenamiento. No toma decisiones de aprobación.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path
from contextlib import closing

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mlops_pipeline"))

from ft_engineering import VARIABLES_BASE, cargar_y_separar_datos
from model_monitoring import (
    CATEGORICAS,
    analizar_drift,
    preparar_serie,
)


st.set_page_config(
    page_title="Monitoreo de créditos",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def cargar_modelo():
    """Cargar únicamente el modelo local generado por el proyecto."""
    return joblib.load(ROOT / "mejor_modelo.joblib")


@st.cache_data
def cargar_referencia():
    """Recuperar la misma separación usada durante entrenamiento."""
    referencia, prueba, _, _ = cargar_y_separar_datos()
    return referencia, prueba


st.title("Análisis y monitoreo de créditos")
st.caption(
    "Proyecto académico de Machine Learning · "
    "Predicciones por lote y cambios en los datos"
)

st.warning(
    "El modelo tiene capacidad limitada para detectar la clase 0. "
    "Las salidas requieren revisión humana. La interpretación de "
    "las clases y la disponibilidad temporal de los campos "
    "siguen pendientes de confirmación."
)

try:
    modelo = cargar_modelo()
    referencia, prueba = cargar_referencia()
except Exception as error:
    st.error(f"No fue posible iniciar la aplicación: {error}")
    st.stop()

archivo = st.file_uploader(
    "Carga un CSV con las ocho entradas del modelo",
    type=["csv"],
)

with st.expander("Columnas requeridas"):
    st.write(VARIABLES_BASE)
    st.write(
        "Pago_atiempo es opcional y solo se utiliza para evaluar "
        "cuando contiene resultados reales observados."
    )

if archivo is None:
    st.info(
        "Modo demostración: se utiliza el conjunto de prueba. "
        "No representa un lote nuevo ni evolución temporal."
    )
    actual = prueba.copy()
else:
    try:
        actual = pd.read_csv(archivo)
    except Exception as error:
        st.error(f"No se pudo leer el CSV: {error}")
        st.stop()

faltantes = set(VARIABLES_BASE) - set(actual.columns)

if actual.empty or faltantes:
    st.error(
        "El lote está vacío o faltan columnas: "
        + ", ".join(sorted(faltantes))
    )
    st.stop()

try:
    # El pipeline aplica las mismas transformaciones del entrenamiento.
    entradas = actual[VARIABLES_BASE].copy()
    predicciones = modelo.predict(entradas)

    indice_clase_0 = list(modelo.classes_).index(0)
    probabilidades = modelo.predict_proba(entradas)[:, indice_clase_0]

    reporte = analizar_drift(referencia, entradas)
except Exception as error:
    st.error(f"No se pudo procesar el lote: {error}")
    st.stop()

salida = actual.copy()
salida["prediccion"] = predicciones
salida["probabilidad_modelo_clase_0"] = probabilidades

alertas = int((reporte["estado"] == "REVISAR").sum())
insuficientes = int(
    (reporte["estado"] == "DATOS_INSUFICIENTES").sum()
)

c1, c2, c3 = st.columns(3)
c1.metric("Registros del lote", len(actual))
c2.metric("Predicciones de clase 0", int((predicciones == 0).sum()))
c3.metric("Variables con alerta", alertas)

pestana_pred, pestana_drift, pestana_hist = st.tabs([
    "Predicciones",
    "Monitoreo de datos",
    "Historial de análisis",
])

with pestana_pred:
    st.subheader("Resultados del lote")
    st.caption(
        "Las probabilidades son estimaciones del modelo y no se han "
        "calibrado como probabilidades de incumplimiento."
    )
    st.dataframe(salida.head(100), width="stretch")

    st.download_button(
        "Descargar todas las predicciones",
        salida.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicciones_lote.csv",
        mime="text/csv",
    )

    # Evaluar únicamente filas que tengan etiquetas reales válidas.
    if "Pago_atiempo" in actual.columns:
        etiquetas = pd.to_numeric(
            actual["Pago_atiempo"], errors="coerce"
        )
        validas = etiquetas.isin([0, 1])

        st.write(
            f"Etiquetas válidas para evaluar: "
            f"{int(validas.sum())} de {len(actual)}."
        )

        if validas.any():
            evaluacion = classification_report(
                etiquetas.loc[validas].astype(int),
                predicciones[validas.to_numpy()],
                labels=[0, 1],
                target_names=["Clase 0", "Clase 1"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(evaluacion).T)
    else:
        st.info(
            "Sin etiquetas reales no puede calcularse el desempeño "
            "del modelo en este lote."
        )

with pestana_drift:
    st.subheader("Comparación con entrenamiento")

    if alertas:
        st.error(
            "Hay cambios que requieren revisión. Comprueba calidad, "
            "origen y población del lote antes de decidir reentrenar."
        )
    elif insuficientes:
        st.warning("Hay variables con datos insuficientes.")
    else:
        st.success("Sin alertas según los umbrales configurados.")

    st.caption(
        "Umbrales exploratorios: PSI ≥ 0,20; "
        "KS ≥ 0,10 con p < 0,01; distancia categórica ≥ 0,10; "
        "cambio de nulos ≥ 10 puntos porcentuales. "
        "No garantizan ausencia de drift."
    )
    st.dataframe(reporte, width="stretch")

    variable = st.selectbox("Variable para comparar", VARIABLES_BASE)
    categorica = variable in CATEGORICAS

    ref = preparar_serie(referencia[variable], categorica)
    act = preparar_serie(entradas[variable], categorica)

    fig, ax = plt.subplots(figsize=(9, 4))

    if categorica:
        proporciones = pd.DataFrame({
            "Referencia": ref.fillna("Sin información").value_counts(
                normalize=True
            ),
            "Actual": act.fillna("Sin información").value_counts(
                normalize=True
            ),
        }).fillna(0)

        proporciones.plot.bar(ax=ax)
        ax.set_ylabel("Proporción")
        ax.tick_params(axis="x", labelrotation=25)
    else:
        # Usar los mismos intervalos y normalizar ambos histogramas.
        valores_ref = ref.dropna().to_numpy(dtype=float)
        valores_act = act.dropna().to_numpy(dtype=float)

        if len(valores_ref) and len(valores_act):
            ax.hist(
                [valores_ref, valores_act],
                bins=30,
                density=True,
                label=["Referencia", "Actual"],
                alpha=0.65,
            )
            ax.set_ylabel("Densidad")
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, "Sin valores válidos para comparar",
                ha="center", transform=ax.transAxes
            )

    ax.set_title(variable)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Descargar reporte de drift",
        reporte.to_csv(index=False).encode("utf-8-sig"),
        file_name="reporte_drift.csv",
        mime="text/csv",
    )

with pestana_hist:
    st.subheader("Historial del monitoreo periódico")

    st.caption(
        "Las fechas corresponden a la ejecución de los análisis. "
        "Los lotes de demostración no representan datos nuevos "
        "ni evolución temporal real."
    )

    # Al pulsar el botón, Streamlit vuelve a consultar la base.
    st.button("Actualizar historial")

    ruta_bd = Path(
        os.environ.get(
            "CREDITOS_HISTORIAL_DB",
            str(ROOT / "monitoreo.sqlite"),
        )
    )

    if not ruta_bd.exists():
        st.info(
            "Todavía no existe historial. Ejecuta primero "
            "model_monitoring.py con un CSV."
        )
    else:
        try:
            # Abrir en modo lectura para no modificar la base desde aquí.
            uri_bd = ruta_bd.resolve().as_uri() + "?mode=ro"

            with closing(sqlite3.connect(uri_bd, uri=True)) as conexion:
                lotes = pd.read_sql_query(
                    """
                    SELECT id, fecha_utc, registros, variables_alerta
                    FROM lotes
                    ORDER BY fecha_utc
                    """,
                    conexion,
                )

                if lotes.empty:
                    st.info("No hay lotes registrados.")
                else:
                    st.dataframe(lotes, width="stretch")

                    # Evolución de las alertas entre lotes guardados.
                    evolucion = lotes.copy()
                    evolucion["fecha_utc"] = pd.to_datetime(
                        evolucion["fecha_utc"], utc=True
                    )

                    st.line_chart(
                        evolucion.set_index("fecha_utc")[
                            ["variables_alerta"]
                        ]
                    )

                    st.caption(
                        "Se necesitan varios lotes distintos para "
                        "observar evolución. Una línea de alertas "
                        "no demuestra por sí sola una tendencia."
                    )

                    lote_id = st.selectbox(
                        "Selecciona un lote para consultar",
                        options=lotes["id"].tolist(),
                        index=len(lotes) - 1,
                    )

                    # Consultas parametrizadas para recuperar el lote.
                    drift_guardado = pd.read_sql_query(
                        """
                        SELECT metricas_json
                        FROM drift
                        WHERE lote_id = ?
                        ORDER BY variable
                        """,
                        conexion,
                        params=(lote_id,),
                    )

                    datos_guardados = pd.read_sql_query(
                        """
                        SELECT datos_json
                        FROM predicciones
                        WHERE lote_id = ?
                        ORDER BY numero_fila
                        """,
                        conexion,
                        params=(lote_id,),
                    )

                    reporte_lote = pd.DataFrame([
                        json.loads(valor)
                        for valor in drift_guardado["metricas_json"]
                    ])

                    predicciones_lote = pd.DataFrame([
                        json.loads(valor)
                        for valor in datos_guardados["datos_json"]
                    ])

                    st.write("Métricas de drift del lote")
                    st.dataframe(
                        reporte_lote,
                        width="stretch",
                    )

                    st.write("Datos y predicciones almacenados")
                    st.dataframe(
                        predicciones_lote.head(100),
                        width="stretch",
                    )

                    st.download_button(
                        "Descargar lote almacenado completo",
                        predicciones_lote.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name=f"predicciones_{lote_id}.csv",
                        mime="text/csv",
                    )

        except (sqlite3.Error, pd.errors.DatabaseError,
                ValueError, KeyError) as error:
            st.error(f"No se pudo consultar el historial: {error}")