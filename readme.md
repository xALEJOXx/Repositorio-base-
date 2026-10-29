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