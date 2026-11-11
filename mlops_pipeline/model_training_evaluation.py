"""
PUNTO 2 - Comparación inicial de modelos.

Se utiliza validación cruzada con cinco particiones de entrenamiento.
La preparación se ajusta nuevamente dentro de cada partición.

Criterio principal: F1 de la clase 0, minoritaria.
La interpretación de 0 como incumplimiento sigue pendiente
de confirmación con la fuente.
"""

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ft_engineering import (
    cargar_y_separar_datos,
    crear_pipeline_preparacion,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

def build_model(clasificador):
    """Une la preparación de datos y el clasificador."""

    return Pipeline([
        ("preprocesamiento", crear_pipeline_preparacion()),
        ("modelo", clasificador),
    ])


def comparar_modelos(X_train, y_train):
    """Compara desempeño, variación y tiempos usando entrenamiento."""

    # El modelo base permite comprobar si los demás aportan valor.
    modelos = {
        "Base_mayoritaria": DummyClassifier(
            strategy="most_frequent",
            random_state=42,
        ),
        "Regresion_logistica": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        ),
        "Arbol_decision": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "Random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Evaluar explícitamente la clase 0.
    # zero_division=0 permite evaluar también la base mayoritaria.
    metricas = {
        "f1_clase_0": make_scorer(
            f1_score, pos_label=0, zero_division=0
        ),
        "precision_clase_0": make_scorer(
            precision_score, pos_label=0, zero_division=0
        ),
        "recall_clase_0": make_scorer(
            recall_score, pos_label=0, zero_division=0
        ),
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    # Usar las mismas particiones para comparar todos los modelos.
    validacion = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    filas = []

    for nombre, clasificador in modelos.items():
        print(f"Evaluando: {nombre}", flush=True)

        resultados = cross_validate(
            estimator=build_model(clasificador),
            X=X_train,
            y=y_train,
            cv=validacion,
            scoring=metricas,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )

        fila = {"modelo": nombre}

        # Promedios y desviaciones entre las cinco particiones.
        for metrica in metricas:
            valores = resultados[f"test_{metrica}"]

            fila[f"{metrica}_cv"] = valores.mean()
            fila[f"{metrica}_std"] = valores.std()

        # Comparar entrenamiento y validación para explorar sobreajuste.
        fila["f1_clase_0_entrenamiento"] = (
            resultados["train_f1_clase_0"].mean()
        )

        fila["brecha_f1"] = (
            fila["f1_clase_0_entrenamiento"]
            - fila["f1_clase_0_cv"]
        )

        # Tiempos orientativos medidos en este computador.
        fila["ajuste_segundos"] = resultados["fit_time"].mean()
        fila["evaluacion_segundos"] = resultados["score_time"].mean()

        filas.append(fila)

    return (
        pd.DataFrame(filas)
        .sort_values("f1_clase_0_cv", ascending=False)
        .reset_index(drop=True)
    )

def summarize_classification(y_real, y_pred):
    """Genera el reporte final con atención a la clase minoritaria."""

    print("\nReporte de clasificación:")
    print(classification_report(
        y_real,
        y_pred,
        labels=[0, 1],
        target_names=["Clase 0", "Clase 1"],
        zero_division=0,
        digits=4,
    ))

    # Filas: clase real. Columnas: clase predicha.
    matriz = pd.DataFrame(
        confusion_matrix(y_real, y_pred, labels=[0, 1]),
        index=["Real 0", "Real 1"],
        columns=["Predicción 0", "Predicción 1"],
    )

    print("\nMatriz de confusión:")
    print(matriz)

    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "precision_clase_0": precision_score(
            y_real, y_pred, pos_label=0, zero_division=0
        ),
        "recall_clase_0": recall_score(
            y_real, y_pred, pos_label=0, zero_division=0
        ),
        "f1_clase_0": f1_score(
            y_real, y_pred, pos_label=0, zero_division=0
        ),
    }

def guardar_resultados(resumen, modelo, metricas, y_real, y_pred):
    """Guarda las evidencias de la comparación y el pipeline elegido."""

    # Guardar las salidas en la raíz, sin agregar carpetas.
    raiz = Path(__file__).resolve().parent.parent

    # Tabla completa de validación cruzada.
    resumen.to_csv(
        raiz / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Métricas finales del único modelo evaluado en prueba.
    with open(
        raiz / "metricas_prueba.json",
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            {nombre: float(valor) for nombre, valor in metricas.items()},
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    # Conservar preparación y clasificador en un solo objeto.
    joblib.dump(modelo, raiz / "mejor_modelo.joblib")

    # Comparar F1 y variación entre particiones.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        resumen["modelo"],
        resumen["f1_clase_0_cv"],
        yerr=resumen["f1_clase_0_std"],
        capsize=5,
        color="steelblue",
    )

    ax.set_title("F1 de clase 0 en validación cruzada")
    ax.set_ylabel("F1 medio ± desviación estándar")
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    fig.savefig(
        raiz / "comparacion_f1.png",
        dpi=150,
    )
    plt.close(fig)

    # Visualizar la brecha entre entrenamiento y validación.
    fig, ax = plt.subplots(figsize=(9, 5))

    resumen.set_index("modelo")[
        ["f1_clase_0_entrenamiento", "f1_clase_0_cv"]
    ].plot.bar(ax=ax)

    ax.set_title("F1 de clase 0: entrenamiento y validación")
    ax.set_ylabel("F1")
    ax.set_xlabel("")
    ax.legend(["Entrenamiento", "Validación"])
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    fig.savefig(
        raiz / "comparacion_sobreajuste.png",
        dpi=150,
    )
    plt.close(fig)

    # Matriz final con filas reales y columnas predichas.
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_pred,
        labels=[0, 1],
        display_labels=["Clase 0", "Clase 1"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("Random Forest: matriz de confusión en prueba")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    fig.tight_layout()
    fig.savefig(
        raiz / "matriz_confusion_prueba.png",
        dpi=150,
    )
    plt.close(fig)

    print("\nResultados, gráficas y pipeline guardados.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_y_separar_datos()

    # X_test e y_test permanecen reservados para la evaluación final.
    print("PUNTO 2 - Comparación mediante validación cruzada")
    print("Criterio principal: F1 de la clase 0.\n")

    resumen = comparar_modelos(X_train, y_train)

    columnas_reporte = [
        "modelo",
        "f1_clase_0_cv",
        "f1_clase_0_std",
        "precision_clase_0_cv",
        "recall_clase_0_cv",
        "balanced_accuracy_cv",
        "accuracy_cv",
        "brecha_f1",
        "ajuste_segundos",
    ]

    print("\nResultados de validación:")
    print(resumen[columnas_reporte].round(4).to_string(index=False))

    print("\nComparación inicial completada.")
    print("El conjunto de prueba todavía no se ha evaluado.")

    # Selección por el criterio definido antes de consultar prueba.
    nombre_seleccionado = resumen.iloc[0]["modelo"]

    print("\nModelo seleccionado por F1 en validación:")
    print(nombre_seleccionado)

    # Mantener exactamente la configuración comparada.
    # Este paso corresponde al resultado actual: Random Forest.
    if nombre_seleccionado != "Random_forest":
        raise ValueError(
            "Cambió el ganador. Revisar su configuración antes de evaluar."
        )

    mejor_modelo = build_model(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    # Ajustar el pipeline completo con todo el entrenamiento.
    mejor_modelo.fit(X_train, y_train)

    # Evaluar únicamente el modelo seleccionado en prueba.
    predicciones = mejor_modelo.predict(X_test)

    metricas_prueba = summarize_classification(
        y_test, predicciones
    )

    print("\nResumen de métricas en prueba:")
    print(pd.Series(metricas_prueba).round(4))

    # El resultado de prueba se reporta; no se usa para elegir
    # otro modelo o ajustar parámetros en esta comparación.

    # Guardar las evidencias correspondientes a esta ejecución.
    guardar_resultados(
        resumen,
        mejor_modelo,
        metricas_prueba,
        y_test,
        predicciones,
    )