# Análisis del cumplimiento de pago de créditos

Proyecto final de Machine Learning.

## Objetivo

Explorar una base histórica de créditos e identificar características
asociadas al cumplimiento de pago, como preparación para desarrollar
posteriormente un modelo de clasificación binaria.

## Alcance de la primera entrega

El punto 1 incluye:

- Organización del repositorio.
- Configuración del entorno virtual y dependencias.
- Carga de datos y generación del CSV.
- Caracterización de las variables.
- Revisión de nulos, duplicados, tipos e inconsistencias.
- Análisis univariable, bivariable y multivariable.
- Interpretación de resultados.
- Propuesta de reglas de validación y transformaciones futuras.

Esta entrega no incluye entrenamiento de modelos ni despliegue.

## Datos utilizados

La base suministrada, BD_creditos (2).xlsx, contiene 10.763 registros
y 23 variables. El notebook de carga genera Base_de_datos.csv
conservando los valores originales.

La entidad de origen y las unidades monetarias están pendientes
de confirmación.

La variable objetivo es Pago_atiempo. Se interpreta provisionalmente:

- 0: no paga a tiempo.
- 1: paga a tiempo.

La codificación y el periodo de pago evaluado deben confirmarse
con la fuente.

## Archivos principales

| Archivo | Función |
|---|---|
| mlops_pipeline/cargar_datos.ipynb | Cargar el Excel y generar el CSV |
| mlops_pipeline/eda_analisis.ipynb | Explorar y analizar los datos |
| requirements.txt | Dependencias del proyecto |
| set_up.bat | Prepara el entorno virtual e instala y verifica las dependencias |
| .gitignore | Archivos y carpetas excluidos del control de versiones |
| Base_de_datos.csv | Base original convertida a CSV |

Los componentes ft_eng.py, model_evaluation.py, deply.py y
monitoring.py están reservados para etapas posteriores.

## Preparación del entorno

Se utilizó Python 3.13 en Windows.

Desde una terminal ubicada en la raíz del repositorio:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En VS Code se requieren las extensiones Python y Jupyter de Microsoft.
En cada notebook debe seleccionarse el kernel del entorno .venv.

## Orden de ejecución

1. Para reproducir la conversión, colocar BD_creditos (2).xlsx
   en la raíz y ejecutar mlops_pipeline/cargar_datos.ipynb.
2. Comprobar que se haya generado Base_de_datos.csv.
3. Ejecutar mlops_pipeline/eda_analisis.ipynb desde la primera celda.

Si Base_de_datos.csv ya está disponible, el análisis exploratorio
puede ejecutarse sin repetir la conversión del Excel.

El CSV conserva los datos originales. Los ajustes del análisis
se realizan en una copia en memoria.

## Hallazgos principales

- No se encontraron filas completamente duplicadas.
- La clase 1 representa el 95,25 % de los registros y la clase 0
  el 4,75 %, por lo que existe un fuerte desbalance.
- Se identificaron 58 valores inesperados en tendencia_ingresos,
  tratados provisionalmente como faltantes.
- Se detectaron 150 edades superiores a 100 años, 24 salarios
  iguales a cero y un puntaje de Datacrédito negativo.
  Estos registros requieren validación y se conservaron.
- La clase 0 presenta medianas mayores de capital prestado
  y cuota pactada que la clase 1.
- La edad mediana es de 38 años en la clase 0 y 42 en la clase 1.
- La mediana del puntaje de Datacrédito es de 768 en la clase 0
  y 792 en la clase 1.
- puntaje presenta la mayor correlación de Spearman con el
  objetivo: 0,639.

## Limitaciones y decisiones pendientes

Es necesario confirmar el significado de los campos, las unidades,
los códigos especiales y la definición temporal del objetivo.

También debe comprobarse que saldos y puntajes estuvieran disponibles
antes del resultado que se desea predecir, para evitar fuga
de información.

Las asociaciones observadas no demuestran causalidad.
Las reglas y transformaciones propuestas se documentan en el EDA;
su implementación para modelado corresponde a una etapa posterior.

## Ramas del proyecto

El repositorio dispone de developer, certification y master.
El desarrollo de esta entrega se realiza en developer.

## Punto 2: ingeniería de características y evaluación

### Preparación de datos

Se seleccionaron ocho características iniciales y se crearon
relacion_cuota_salario y salario_cero.

La selección es provisional y supone que los campos se conocen
al otorgar el crédito. Los demás datos se excluyeron de esta versión
hasta confirmar su origen y disponibilidad temporal.

Se aplicaron reglas de limpieza y un pipeline con:

- Imputación por mediana para variables numéricas.
- Escalamiento de variables numéricas.
- Categoría de información faltante para variables categóricas.
- Codificación One-Hot con manejo de categorías desconocidas.

El tratamiento produce 16 columnas en el ajuste con todo el
entrenamiento. Las categorías disponibles pueden variar entre
particiones de validación.

### Separación y comparación

Se reservaron 8.610 registros para entrenamiento y 2.153 para prueba,
mediante una división estratificada con semilla 42.

Los modelos se compararon mediante validación cruzada estratificada
de cinco particiones dentro del entrenamiento. El pipeline se ajustó
de nuevo en cada partición para evitar fuga durante la preparación.

Se compararon regresión logística, árbol de decisión y Random Forest,
junto con una base que siempre predice la clase mayoritaria.

El criterio principal fue el F1 de la clase 0. Se conservaron las
etiquetas originales; interpretar 0 como incumplimiento requiere
confirmación con la fuente.

### Resultados de validación

| Modelo | F1 clase 0 | Desviación F1 | Precisión clase 0 | Recall clase 0 |
|---|---:|---:|---:|---:|
| Random Forest | 0,1361 | 0,0195 | 0,0960 | 0,2348 |
| Árbol de decisión | 0,1120 | 0,0071 | 0,0632 | 0,4988 |
| Regresión logística | 0,1080 | 0,0078 | 0,0607 | 0,4890 |
| Base mayoritaria | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

Random Forest obtuvo el mayor F1 medio y se seleccionó con ese criterio.
Sin embargo, presentó una brecha de F1 entre entrenamiento y validación
de 0,3622, señal de sobreajuste.

El árbol y la regresión logística detectaron más casos de clase 0,
pero generaron una mayor proporción de alertas incorrectas.

### Evaluación final en prueba

El modelo seleccionado se ajustó con todo el entrenamiento y se
evaluó en el conjunto de prueba reservado.

| Métrica | Resultado |
|---|---:|
| Exactitud | 0,8439 |
| Exactitud balanceada | 0,5780 |
| Precisión clase 0 | 0,0993 |
| Recall clase 0 | 0,2843 |
| F1 clase 0 | 0,1472 |

De los 102 casos reales de clase 0, identificó 29 y omitió 73.
Además, clasificó incorrectamente 263 casos de clase 1 como clase 0.

El resultado evidencia una capacidad limitada para detectar la clase
minoritaria. La selección representa el mejor F1 entre las
configuraciones evaluadas, no un modelo listo para uso operativo.

### Consistencia y costo computacional

La desviación del F1 entre particiones se utilizó para describir
la variación del desempeño.

Se registraron tiempos de ajuste y evaluación en comparacion_modelos.csv.
Random Forest requirió más tiempo de ajuste que el árbol y la
regresión logística en este equipo.

Estos tiempos corresponden a la base actual y no constituyen una
prueba de escalabilidad. No se realizaron ensayos con mayores
volúmenes ni cargas simultáneas.

### Evidencias generadas

- comparacion_modelos.csv: métricas y tiempos de validación.
- metricas_prueba.json: métricas finales del seleccionado.
- mejor_modelo.joblib: pipeline de preparación y clasificador.
- comparacion_f1.png: comparación de F1 en validación.
- comparacion_sobreajuste.png: F1 de entrenamiento y validación.
- matriz_confusion_prueba.png: resultados en prueba.

### Ejecución del punto 2

Desde la raíz del repositorio, con el entorno virtual activo:

python mlops_pipeline/ft_eng.py
python mlops_pipeline/model_evaluation.py

El pipeline guardado requiere conservar el módulo ft_eng.py y
utilizar un entorno compatible con las versiones del entrenamiento.

### Limitaciones

La división aleatoria no demuestra desempeño sobre créditos futuros.
Tampoco garantiza separación por cliente, pues no se dispone de un
identificador confirmado.

El EDA previo utilizó la base completa, por lo que la prueba no es
totalmente independiente de las decisiones exploratorias. Se requeriría
una muestra nueva o temporal para una evaluación más rigurosa.

Las reglas de edad y el significado de las variables y del objetivo
continúan sujetos a validación con la fuente.