"""
Monitoreo de datos y predicciones de créditos.

- Numéricas: KS y PSI.
- Categóricas: distancia de variación total.
- Calidad: cambios en la proporción de nulos.
- Persistencia: lotes, predicciones y métricas en SQLite.
- Periodicidad: revisión del CSV según el intervalo configurado.

El drift indica cambios en los datos, no demuestra por sí solo
una pérdida de precisión del modelo.
"""

import argparse
import hashlib
import json
import sqlite3
import time
import uuid
from io import BytesIO
from pathlib import Path
from contextlib import closing

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ft_engineering import VARIABLES_BASE, cargar_y_separar_datos


ROOT = Path(__file__).resolve().parent.parent
CATEGORICAS = ["tipo_credito", "tipo_laboral"]


def calcular_psi(referencia, actual):
    """Compara proporciones en intervalos definidos por referencia."""

    referencia = np.asarray(referencia, dtype=float)
    actual = np.asarray(actual, dtype=float)

    referencia = referencia[np.isfinite(referencia)]
    actual = actual[np.isfinite(actual)]

    if not len(referencia) or not len(actual):
        return np.nan

    cortes = np.unique(
        np.quantile(referencia, np.linspace(0, 1, 11))
    )

    if len(cortes) == 1:
        # Diferenciar valores inferiores, iguales y superiores
        # cuando la referencia es constante.
        valor = cortes[0]
        limites = np.unique([
            -np.inf,
            valor,
            np.nextafter(valor, np.inf),
            np.inf,
        ])
    else:
        # Con pocas cantidades distintas, separar sus valores.
        # Con más cantidades, utilizar cortes por cuantiles.
        if len(cortes) <= 3:
            interiores = cortes[:-1] + np.diff(cortes) / 2
        else:
            interiores = cortes[1:-1]

        limites = np.r_[-np.inf, interiores, np.inf]

    conteo_ref = np.histogram(referencia, bins=limites)[0]
    conteo_actual = np.histogram(actual, bins=limites)[0]

    # Suavizar para evitar divisiones y logaritmos de cero.
    p = (conteo_ref + 0.5) / (
        len(referencia) + 0.5 * len(conteo_ref)
    )
    q = (conteo_actual + 0.5) / (
        len(actual) + 0.5 * len(conteo_actual)
    )

    return float(np.sum((q - p) * np.log(q / p)))


def preparar_serie(serie, categorica):
    """Uniforma faltantes y convierte cantidades a valores finitos."""

    texto = serie.astype("string").str.strip()

    texto = texto.mask(
        texto.str.lower().isin(
            ["", "na", "n/a", "null", "none", "nan"]
        )
    )

    if categorica:
        return texto

    return pd.to_numeric(texto, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def analizar_drift(referencia, actual):
    """Devuelve métricas y alertas por entrada del modelo."""

    faltantes = set(VARIABLES_BASE) - (
        set(referencia.columns) & set(actual.columns)
    )

    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    if referencia.empty or actual.empty:
        raise ValueError("Ambos conjuntos deben contener registros.")

    filas = []

    for columna in VARIABLES_BASE:
        categorica = columna in CATEGORICAS
        ref = preparar_serie(referencia[columna], categorica)
        act = preparar_serie(actual[columna], categorica)

        diferencia_nulos = abs(
            float(act.isna().mean()) - float(ref.isna().mean())
        )

        fila = {
            "variable": columna,
            "tipo": "categorica" if categorica else "numerica",
            "registros_referencia": len(ref),
            "registros_actuales": len(act),
            "nulos_referencia": float(ref.isna().mean()),
            "nulos_actuales": float(act.isna().mean()),
            "diferencia_nulos": diferencia_nulos,
            "ks": np.nan,
            "p_valor_ks": np.nan,
            "psi": np.nan,
            "distancia_categorica": np.nan,
        }

        alerta_nulos = diferencia_nulos >= 0.10

        if categorica:
            # Incluir los faltantes como grupo en la comparación.
            ref = ref.fillna("__FALTANTE__")
            act = act.fillna("__FALTANTE__")

            categorias = sorted(set(ref).union(act))

            p = ref.value_counts(normalize=True).reindex(
                categorias, fill_value=0
            )
            q = act.value_counts(normalize=True).reindex(
                categorias, fill_value=0
            )

            distancia = float(0.5 * np.abs(p - q).sum())
            fila["distancia_categorica"] = distancia
            alerta_distribucion = distancia >= 0.10
            suficientes = min(len(ref), len(act)) >= 20

        else:
            ref_validos = ref.dropna().to_numpy(dtype=float)
            act_validos = act.dropna().to_numpy(dtype=float)
            suficientes = min(
                len(ref_validos), len(act_validos)
            ) >= 20

            alerta_distribucion = False

            if suficientes:
                ks, p_valor = ks_2samp(
                    ref_validos, act_validos
                )
                psi = calcular_psi(ref_validos, act_validos)

                fila.update({
                    "ks": float(ks),
                    "p_valor_ks": float(p_valor),
                    "psi": psi,
                })

                # Considerar significancia y tamaño del cambio.
                alerta_distribucion = (
                    (p_valor < 0.01 and ks >= 0.10)
                    or psi >= 0.20
                )

        if alerta_nulos:
            fila["estado"] = "REVISAR"
        elif not suficientes:
            fila["estado"] = "DATOS_INSUFICIENTES"
        elif alerta_distribucion:
            fila["estado"] = "REVISAR"
        else:
            fila["estado"] = "SIN_ALERTA"

        filas.append(fila)

    return pd.DataFrame(filas)


def crear_tablas(conexion):
    """Prepara las tablas locales sin borrar datos existentes."""

    conexion.execute("""
        CREATE TABLE IF NOT EXISTS lotes (
            id TEXT PRIMARY KEY,
            firma TEXT UNIQUE NOT NULL,
            fecha_utc TEXT NOT NULL,
            registros INTEGER NOT NULL,
            variables_alerta INTEGER NOT NULL
        )
    """)

    conexion.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            lote_id TEXT NOT NULL,
            numero_fila INTEGER NOT NULL,
            datos_json TEXT NOT NULL,
            PRIMARY KEY (lote_id, numero_fila)
        )
    """)

    conexion.execute("""
        CREATE TABLE IF NOT EXISTS drift (
            lote_id TEXT NOT NULL,
            variable TEXT NOT NULL,
            estado TEXT NOT NULL,
            metricas_json TEXT NOT NULL,
            PRIMARY KEY (lote_id, variable)
        )
    """)


def procesar_y_guardar_lote(ruta_actual, referencia):
    """Valida el CSV y guarda sus predicciones y métricas."""

    # Solo permitir archivos CSV dentro de la carpeta del proyecto.
    # resolve() también resuelve ".." y enlaces simbólicos.
    carpeta_permitida = ROOT.resolve()
    ruta_actual = Path(ruta_actual)

    if not ruta_actual.is_absolute():
        ruta_actual = carpeta_permitida / ruta_actual

    ruta_actual = ruta_actual.resolve()

    if not ruta_actual.is_relative_to(carpeta_permitida):
        raise ValueError(
            "El lote debe estar dentro de la carpeta del proyecto."
        )

    if ruta_actual.suffix.lower() != ".csv":
        raise ValueError("El lote debe ser un archivo CSV.")

    if not ruta_actual.is_file():
        raise ValueError("El archivo CSV indicado no existe.")

    ruta_bd = carpeta_permitida / "monitoreo.sqlite"

    # Leer una sola versión del lote y del modelo.
    contenido = ruta_actual.read_bytes()
    contenido_modelo = (ROOT / "mejor_modelo.joblib").read_bytes()

    # Reconocer contenido ya procesado con el mismo modelo y referencia.
    firma = hashlib.sha256()
    firma.update(contenido)
    firma.update(contenido_modelo)
    firma.update(
        referencia[VARIABLES_BASE].to_csv(
            index=False
        ).encode("utf-8")
    )
    identificador = firma.hexdigest()

    with closing(sqlite3.connect(ruta_bd, timeout=30)) as conexion, conexion:
        crear_tablas(conexion)

        existente = conexion.execute(
            "SELECT id FROM lotes WHERE firma = ?",
            (identificador,),
        ).fetchone()

    if existente:
        print("Lote sin cambios: ya fue analizado.", flush=True)
        return

    actual = pd.read_csv(BytesIO(contenido))

    # Validar las columnas y obtener las métricas.
    reporte = analizar_drift(referencia, actual)

    # Cargar únicamente el modelo local generado por el proyecto.
    modelo = joblib.load(BytesIO(contenido_modelo))
    entradas = actual[VARIABLES_BASE]

    predicciones = modelo.predict(entradas)
    posicion_clase_0 = list(modelo.classes_).index(0)
    probabilidades = modelo.predict_proba(
        entradas
    )[:, posicion_clase_0]

    salida = entradas.copy()

    if "Pago_atiempo" in actual.columns:
        salida["Pago_atiempo"] = actual["Pago_atiempo"]

    salida["prediccion"] = predicciones
    salida["probabilidad_clase_0"] = probabilidades

    # La conversión de pandas representa los faltantes como null.
    filas_datos = json.loads(
        salida.to_json(orient="records", date_format="iso")
    )
    filas_drift = json.loads(
        reporte.to_json(orient="records")
    )

    lote_id = str(uuid.uuid4())
    fecha = pd.Timestamp.now(tz="UTC").isoformat()
    alertas = int((reporte["estado"] == "REVISAR").sum())

    with closing(sqlite3.connect(ruta_bd, timeout=30)) as conexion, conexion:
        # Bloquear escrituras concurrentes y comprobar duplicados otra vez.
        conexion.execute("BEGIN IMMEDIATE")

        existente = conexion.execute(
            "SELECT id FROM lotes WHERE firma = ?",
            (identificador,),
        ).fetchone()

        if existente:
            print("Lote ya registrado por otro proceso.", flush=True)
            return

        # Todo el lote se guarda en una misma transacción.
        conexion.execute(
            """
            INSERT INTO lotes
            (id, firma, fecha_utc, registros, variables_alerta)
            VALUES (?, ?, ?, ?, ?)
            """,
            (lote_id, identificador, fecha, len(actual), alertas),
        )

        conexion.executemany(
            """
            INSERT INTO predicciones
            (lote_id, numero_fila, datos_json)
            VALUES (?, ?, ?)
            """,
            [
                (
                    lote_id,
                    numero,
                    json.dumps(fila, ensure_ascii=False),
                )
                for numero, fila in enumerate(filas_datos)
            ],
        )

        conexion.executemany(
            """
            INSERT INTO drift
            (lote_id, variable, estado, metricas_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    lote_id,
                    fila["variable"],
                    fila["estado"],
                    json.dumps(fila, ensure_ascii=False),
                )
                for fila in filas_drift
            ],
        )

    print(
        f"Lote guardado: {len(actual)} registros; "
        f"{alertas} variables con alerta.",
        flush=True,
    )

    if alertas:
        print(
            "ALERTA: revisar calidad, origen y población. "
            "Evaluar desempeño con etiquetas antes de reentrenar.",
            flush=True,
        )

    print(
        "Registro disponible en monitoreo.sqlite. "
        "Sin alertas no garantiza ausencia de cambios.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Monitoreo de lotes de créditos."
    )
    parser.add_argument(
        "--actual",
        help="Ruta del CSV que se desea monitorear.",
    )
    parser.add_argument(
        "--intervalo",
        type=int,
        default=0,
        help="Segundos entre revisiones; cero ejecuta una sola vez.",
    )
    args = parser.parse_args()

    if args.intervalo < 0:
        parser.error("El intervalo no puede ser negativo.")

    if args.intervalo and not args.actual:
        parser.error("El monitoreo periódico requiere --actual.")

    referencia, prueba, _, _ = cargar_y_separar_datos()

    # La demostración no se incorpora al historial automáticamente.
    if not args.actual:
        print(
            "DEMOSTRACIÓN: entrenamiento frente a prueba.\n"
            "No representa evolución temporal."
        )

        reporte = analizar_drift(referencia, prueba)
        reporte.to_csv(
            ROOT / "reporte_drift.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(reporte.round(4).to_string(index=False))
        print("\nReporte guardado en reporte_drift.csv")
        return

    print(
        "Los CSV históricos o simulados deben documentarse "
        "como demostraciones.",
        flush=True,
    )

    try:
        while True:
            try:
                procesar_y_guardar_lote(
                    args.actual, referencia
                )
            except Exception as error:
                if args.intervalo == 0:
                    raise

                # Un error temporal no detiene las siguientes revisiones.
                print(
                    f"No se pudo procesar el lote: {error}",
                    flush=True,
                )

            if args.intervalo == 0:
                break

            time.sleep(args.intervalo)

    except KeyboardInterrupt:
        print("\nMonitoreo detenido por el usuario.")


if __name__ == "__main__":
    main()