"""
PUNTO 2 - Ingeniería de características.

Paso 1: cargar la base, seleccionar las variables iniciales y
separar los datos de entrenamiento y prueba.

La limpieza y las transformaciones se incorporarán posteriormente
en un pipeline.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

# Ubicar el CSV desde la raíz del proyecto.
ROOT = Path(__file__).resolve().parent.parent
RUTA_DATOS = ROOT / "Base_de_datos.csv"

# Selección inicial de características.
# Se supone que estos datos se conocen al otorgar el crédito;
# esta disponibilidad debe confirmarse con la fuente.
VARIABLES_BASE = [
    "tipo_credito",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
]

# Los puntajes y datos de saldos e historial se dejan fuera
# de esta primera versión mientras se confirma su origen
# y momento de medición, para reducir el riesgo de fuga.

def preparar_variables(X):
    """Limpia los campos y crea características sin aprender de los datos."""

    # Trabajar sobre una copia para conservar la entrada original.
    datos = X[VARIABLES_BASE].copy()

    categoricas = ["tipo_credito", "tipo_laboral"]
    numericas = [
        columna for columna in VARIABLES_BASE
        if columna not in categoricas
    ]

    # Convertir cantidades a números; entradas inválidas pasan a nulo.
    for columna in numericas:
        datos[columna] = pd.to_numeric(
            datos[columna], errors="coerce"
        )

    datos[numericas] = datos[numericas].replace(
        [np.inf, -np.inf], np.nan
    )

    # Capital, plazo y cuota deben ser positivos para este ejercicio.
    for columna in ["capital_prestado", "plazo_meses", "cuota_pactada"]:
        datos.loc[datos[columna] <= 0, columna] = np.nan

    # Los montos negativos se consideran pendientes de corrección.
    for columna in ["salario_cliente", "total_otros_prestamos"]:
        datos.loc[datos[columna] < 0, columna] = np.nan

    # Regla provisional: edades fuera de 18 a 100 pasan a nulo.
    # No se eliminan filas. El rango debe validarse con la entidad.
    datos.loc[
        ~datos["edad_cliente"].between(18, 100),
        "edad_cliente"
    ] = np.nan

    # Uniformar las categorías y los marcadores de información faltante.
    for columna in categoricas:
        valores = datos[columna].astype("string").str.strip()

        valores = valores.mask(
            valores.str.lower().isin(
                ["", "na", "n/a", "null", "none", "nan"]
            )
        )

        # Usar cadenas y np.nan compatibles con el futuro imputador.
        datos[columna] = valores.astype(object).where(
            valores.notna(), np.nan
        )

    # Un salario cero se conserva, pero no se utiliza como divisor.
    salario_valido = datos["salario_cliente"].where(
        datos["salario_cliente"] > 0
    )

    # Característica 1: peso de la cuota respecto al salario reportado.
    # Interpretarla como carga mensual requiere confirmar periodicidades.
    datos["relacion_cuota_salario"] = (
        datos["cuota_pactada"] / salario_valido
    )

    # Característica 2: señalar salarios cero, sin confundirlos con nulos.
    datos["salario_cero"] = (
        datos["salario_cliente"] == 0
    ).astype(int)

    return datos


def cargar_y_separar_datos():
    """Devuelve X_train, X_test, y_train e y_test sin transformar."""

    # 1. Leer los datos originales generados en el punto 1.
    datos = pd.read_csv(RUTA_DATOS)

    # 2. Verificar que estén las columnas necesarias.
    requeridas = VARIABLES_BASE + ["Pago_atiempo"]
    faltantes = set(requeridas) - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    # 3. Validar la variable objetivo.
    if datos["Pago_atiempo"].isna().any():
        raise ValueError("Pago_atiempo contiene valores nulos.")

    if not datos["Pago_atiempo"].isin([0, 1]).all():
        raise ValueError("Pago_atiempo debe contener solo 0 y 1.")

    # X contiene las características; y contiene el resultado.
    # Se conserva la codificación original de Pago_atiempo.
    X = datos[VARIABLES_BASE].copy()
    y = datos["Pago_atiempo"].astype(int)

    # 4. Reservar 20 % para la evaluación final.
    # La estratificación conserva aproximadamente la proporción
    # de clases y la semilla permite reproducir la separación.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Las transformaciones se ajustarán después solo con entrenamiento.
    # La comparación de modelos usará validación dentro de entrenamiento.
    return X_train, X_test, y_train, y_test

def crear_pipeline_preparacion():
    """Construye el tratamiento de datos, inicialmente sin ajustar."""

    columnas_numericas = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "relacion_cuota_salario",
        "salario_cero",
    ]

    columnas_categoricas = [
        "tipo_credito",
        "tipo_laboral",
    ]

    # Rellenar nulos numéricos con la mediana y ajustar las escalas.
    pipeline_numerico = Pipeline([
        ("imputacion", SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )),
        ("escalamiento", StandardScaler()),
    ])

    # Representar faltantes y convertir categorías en columnas binarias.
    # Las categorías nuevas se permiten sin provocar un error.
    pipeline_categorico = Pipeline([
        ("imputacion", SimpleImputer(
            strategy="constant",
            fill_value="Sin_informacion",
            keep_empty_features=True,
        )),
        ("codificacion", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    # Aplicar a cada grupo el tratamiento correspondiente.
    tratamiento_columnas = ColumnTransformer([
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas),
    ])

    # Incluir también las reglas de limpieza y variables derivadas.
    return Pipeline([
        ("preparacion", FunctionTransformer(
            preparar_variables,
            validate=False,
        )),
        ("tratamiento", tratamiento_columnas),
    ])

# Ejecutar las comprobaciones al lanzar este archivo.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_y_separar_datos()

    print("PUNTO 2 - Carga y separación de datos")
    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    # Verificar la cantidad de registros de cada clase.
    distribucion = pd.DataFrame({
        "entrenamiento": y_train.value_counts().sort_index(),
        "prueba": y_test.value_counts().sort_index(),
    })

    print("\nCantidad de registros por clase:")
    print(distribucion)

    print("\nPaso 1 completado correctamente.")

    # Aplicar la limpieza y crear las dos variables derivadas.
    # Esta comprobación utiliza únicamente entrenamiento.
    muestra_preparada = preparar_variables(X_train)

    print("\nEntrenamiento con variables derivadas:")
    print(muestra_preparada.shape)

    print("\nValores faltantes después de las reglas de limpieza:")
    print(muestra_preparada.isna().sum())

    print("\nPaso 2: limpieza y variables derivadas verificadas.")

    # Comprobar el pipeline ajustándolo solo con entrenamiento.
    pipeline = crear_pipeline_preparacion()
    entrenamiento_transformado = pipeline.fit_transform(X_train)

    print("\nDimensiones después del pipeline:")
    print(entrenamiento_transformado.shape)

    # La salida debe ser completamente numérica y finita.
    assert np.isfinite(entrenamiento_transformado).all(), (
        "La salida contiene nulos o valores infinitos."
    )

    print("Sin nulos ni infinitos después del tratamiento.")
    print("Paso 3: pipeline de preparación verificado.")