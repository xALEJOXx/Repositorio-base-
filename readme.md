# Análisis del cumplimiento de pago de créditos

Proyecto final de Machine Learning desarrollado en Python. Incluye análisis exploratorio, preparación de variables, comparación de modelos, predicciones por lotes, monitoreo de datos y despliegue mediante una API y Docker.

## Objetivo

Desarrollar un flujo de Machine Learning para explorar las características asociadas al cumplimiento de pago de créditos, comparar modelos supervisados y disponibilizar sus predicciones.

El proyecto incorpora una aplicación Streamlit para consultar resultados y un proceso periódico para detectar cambios en los datos.

## Contexto y datos

Una entidad que otorga créditos necesita identificar características asociadas al cumplimiento de pago y realizar seguimiento a cambios en la población analizada.

La base suministrada, `BD_creditos (2).xlsx`, contiene **10.763 registros y 23 variables**. El notebook de carga genera `Base_de_datos.csv`, conservando los valores originales.

La entidad de origen, las unidades monetarias y el periodo observado están pendientes de confirmación.

### Variable objetivo

La variable objetivo es `Pago_atiempo`. Se conserva su codificación original:

- `0`: interpretado provisionalmente como no pagar a tiempo.
- `1`: interpretado provisionalmente como pagar a tiempo.

Esta interpretación y el periodo de pago evaluado deben confirmarse con la fuente. El problema se aborda como clasificación binaria supervisada.

## Organización del proyecto

| Archivo | Función |
|---|---|
| `mlops_pipeline/Cargar_datos.ipynb` | Cargar el Excel y generar el CSV |
| `mlops_pipeline/comprension_eda.ipynb` | Explorar y caracterizar los datos |
| `mlops_pipeline/ft_engineering.py` | Seleccionar variables, separar conjuntos y construir el pipeline |
| `mlops_pipeline/model_training_evaluation.py` | Comparar modelos, evaluar y guardar el seleccionado |
| `mlops_pipeline/model_monitoring.py` | Detectar drift y registrar lotes y predicciones |
| `mlops_pipeline/model_deploy.py` | Exponer la API de predicción |
| `Base_de_datos.csv` | Base original convertida a CSV |
| `app.py` | Aplicación Streamlit |
| `setup.bat` | Preparación del entorno en Windows |
| `requirements.txt` | Dependencias registradas del entorno de desarrollo |
| `requirements-runtime.txt` | Dependencias directas de ejecución |
| `requirements-runtime.lock` | Dependencias bloqueadas con hashes para Docker |
| `Dockerfile` | Construcción de la imagen de la API |
| `.dockerignore` | Exclusiones del contexto de construcción |
| `test_proyecto.py` | Pruebas automatizadas |
| `.coveragerc` | Configuración de cobertura |
| `sonar-project.properties` | Configuración del análisis estático |
| `.github/workflows/sonar.yml` | Pruebas y análisis desde GitHub Actions |
| `.gitignore` | Exclusiones del control de versiones |
| `readme.md` | Documentación del proyecto |

El nombre `ft_engineering.py` normaliza los espacios inconsistentes que aparecen en ese nombre dentro de la guía.

### Ramas

El repositorio dispone de las ramas `developer`, `certification` y `master`. También conserva `main`.

El desarrollo y la configuración actual de GitHub Actions se realizan en `developer`. La existencia de las ramas no implica que todas contengan los mismos avances.

## Preparación del entorno

Se utilizó Python 3.13 en Windows y un entorno virtual local.

Desde la raíz del proyecto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

También puede ejecutarse el archivo de preparación:

```powershell
.\setup.bat
```

Para asegurar la disponibilidad de las dependencias de ejecución y pruebas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe -m pip install pytest pytest-cov httpx
```

En VS Code se utilizan las extensiones Python y Jupyter de Microsoft. Cada notebook debe ejecutarse con el kernel del entorno `.venv`.

Los comandos siguientes se ejecutan desde la raíz del repositorio.

## Punto 1: carga y comprensión de datos

El primer punto comprende la configuración del repositorio y el entorno, la carga del dataset y el análisis exploratorio.

### Orden de ejecución

1. Para reproducir la conversión, colocar `BD_creditos (2).xlsx` en la raíz.
2. Ejecutar `mlops_pipeline/Cargar_datos.ipynb`.
3. Comprobar la generación de `Base_de_datos.csv`.
4. Ejecutar `mlops_pipeline/comprension_eda.ipynb` desde la primera celda.

Si el CSV ya está disponible, el EDA puede ejecutarse sin repetir la conversión del Excel.

El CSV conserva los datos originales. Los ajustes del EDA se realizan sobre una copia en memoria.

### Actividades realizadas

- Caracterización de variables numéricas, categóricas y temporales.
- Revisión y unificación de representaciones de nulos.
- Ajuste de tipos de datos.
- Revisión de duplicados e inconsistencias.
- Análisis univariable, bivariable y multivariable.
- Interpretación de tablas y gráficos.
- Propuesta de reglas de validación y transformaciones posteriores.

### Hallazgos principales

- No se encontraron filas completamente duplicadas.
- La clase 1 representa el 95,25 % de los registros y la clase 0 el 4,75 %.
- Se identificaron 58 valores inesperados en `tendencia_ingresos`, tratados provisionalmente como faltantes.
- Se detectaron 150 edades superiores a 100 años, 24 salarios iguales a cero y un puntaje de Datacrédito negativo.
- Estos registros se conservaron en el EDA para documentar su revisión.
- La clase 0 presenta medianas mayores de capital prestado y cuota pactada.
- La edad mediana es de 38 años en la clase 0 y 42 años en la clase 1.
- La mediana de `puntaje_datacredito` es de 768 en la clase 0 y 792 en la clase 1.
- `puntaje` presenta la mayor correlación de Spearman con el objetivo: 0,639.

La correlación no demuestra causalidad ni confirma por sí sola utilidad predictiva. En particular, debe verificarse cómo se calculan los puntajes y cuándo están disponibles para descartar fuga de información.

## Avance 1: ingeniería de características y modelos

### Selección de entradas

Se utilizaron ocho características iniciales:

- `tipo_credito`
- `capital_prestado`
- `plazo_meses`
- `edad_cliente`
- `tipo_laboral`
- `salario_cliente`
- `total_otros_prestamos`
- `cuota_pactada`

La selección supone que estos campos se conocen al otorgar el crédito. Esta disponibilidad debe confirmarse con la fuente.

Los demás campos se excluyeron de esta versión mientras se confirma su significado, origen y momento de medición.

### Limpieza y variables derivadas

La preparación incluye:

- Conversión de cantidades a valores numéricos.
- Tratamiento de valores infinitos y entradas no interpretables como faltantes.
- Revisión de cantidades que incumplen las reglas de positividad.
- Regla provisional de edad entre 18 y 100 años.
- Uniformidad de categorías y marcadores de información faltante.

Las edades fuera del rango provisional pasan a nulo durante la preparación para modelado, sin eliminar filas.

Se crearon dos características:

| Característica | Descripción |
|---|---|
| `relacion_cuota_salario` | Cociente entre cuota y salario positivo |
| `salario_cero` | Indicador de salario reportado igual a cero |

La relación cuota/salario no se calcula cuando el salario no es positivo. Interpretarla como carga mensual requiere confirmar la periodicidad de ambos montos.

### Pipeline

El pipeline reúne la limpieza, las variables derivadas y los siguientes tratamientos:

- Imputación por mediana para variables numéricas.
- Escalamiento mediante `StandardScaler`.
- Categoría de información desconocida para faltantes categóricos.
- Codificación One-Hot con manejo de categorías nuevas.

En el ajuste con todo el entrenamiento se obtuvieron 16 columnas. La cantidad puede variar según las categorías presentes en cada partición.

### Separación y validación

Se realizó una división estratificada con semilla 42:

| Conjunto | Clase 0 | Clase 1 | Total |
|---|---:|---:|---:|
| Entrenamiento | 409 | 8.201 | 8.610 |
| Prueba | 102 | 2.051 | 2.153 |

Los modelos se compararon mediante validación cruzada estratificada de cinco particiones dentro del entrenamiento.

El pipeline se ajustó nuevamente en cada partición para evitar fuga durante la preparación. El conjunto de prueba se reservó para evaluar el modelo seleccionado.

Se implementaron las funciones reutilizables `build_model` y `summarize_classification`.

### Modelos y criterio de selección

Se compararon:

- Regresión logística.
- Árbol de decisión.
- Random Forest.
- Clasificador base que siempre predice la clase mayoritaria.

El criterio principal fue el **F1 de la clase 0**, debido a su baja representación. También se registraron precisión, recall, exactitud balanceada, variación entre particiones y tiempos.

### Resultados de validación

| Modelo | F1 clase 0 | Desviación F1 | Precisión clase 0 | Recall clase 0 |
|---|---:|---:|---:|---:|
| Random Forest | 0,1361 | 0,0195 | 0,0960 | 0,2348 |
| Árbol de decisión | 0,1120 | 0,0071 | 0,0632 | 0,4988 |
| Regresión logística | 0,1080 | 0,0078 | 0,0607 | 0,4890 |
| Base mayoritaria | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

Random Forest obtuvo el mayor F1 medio y se seleccionó con ese criterio. Sin embargo, presentó una brecha de F1 entre entrenamiento y validación de 0,3622, señal de sobreajuste.

El árbol y la regresión logística detectaron más casos de clase 0, pero una menor proporción de sus alertas fue correcta.

La base mayoritaria obtuvo aproximadamente 95,25 % de exactitud sin identificar ningún caso de clase 0. Esto evidencia por qué la exactitud por sí sola resulta insuficiente.

### Evaluación final en prueba

| Métrica | Resultado |
|---|---:|
| Exactitud | 0,8439 |
| Exactitud balanceada | 0,5780 |
| Precisión clase 0 | 0,0993 |
| Recall clase 0 | 0,2843 |
| F1 clase 0 | 0,1472 |

Matriz de confusión:

| Clase real | Predicción 0 | Predicción 1 |
|---|---:|---:|
| Clase 0 | 29 | 73 |
| Clase 1 | 263 | 1.788 |

El modelo identificó 29 de los 102 casos reales de clase 0 y omitió 73. Además, clasificó 263 casos de clase 1 como clase 0.

Su capacidad para detectar la clase minoritaria es limitada. La selección corresponde al mayor F1 entre las configuraciones evaluadas y no demuestra que el modelo esté listo para uso operativo.

### Consistencia y costo computacional

Se registraron la desviación del F1 entre particiones, la brecha entre entrenamiento y validación y los tiempos de ajuste y evaluación.

Random Forest requirió más tiempo de ajuste que el árbol y la regresión logística en el equipo utilizado.

Los tiempos corresponden a la base actual. No se realizaron pruebas de escalabilidad con mayores volúmenes ni cargas simultáneas.

### Evidencias

| Archivo | Contenido |
|---|---|
| `comparacion_modelos.csv` | Métricas y tiempos de validación |
| `metricas_prueba.json` | Métricas finales del seleccionado |
| `mejor_modelo.joblib` | Preparación y clasificador en un pipeline |
| `comparacion_f1.png` | Comparación del F1 en validación |
| `comparacion_sobreajuste.png` | Comparación entre entrenamiento y validación |
| `matriz_confusion_prueba.png` | Matriz de confusión en prueba |

### Ejecución

```powershell
.\.venv\Scripts\python.exe .\mlops_pipeline\ft_engineering.py
.\.venv\Scripts\python.exe .\mlops_pipeline\model_training_evaluation.py
```

La segunda instrucción vuelve a entrenar y actualiza el modelo y las evidencias. No es necesario ejecutarla cada vez que se inicia la aplicación.

El modelo guardado requiere conservar `ft_engineering.py` y un entorno compatible con las versiones utilizadas para entrenarlo.

## Avance 2: monitoreo y aplicación Streamlit

### Monitoreo de datos

`model_monitoring.py` compara las entradas de un lote con los datos de entrenamiento utilizados como referencia.

Se calculan:

- KS y PSI para variables numéricas.
- Distancia de variación total para variables categóricas.
- Cambios en la proporción de nulos.

Los valores no interpretables como números también se consideran faltantes para esta revisión.

### Reglas exploratorias de alerta

| Indicador | Condición de revisión |
|---|---|
| PSI | Mayor o igual a 0,20 |
| KS | Estadístico mayor o igual a 0,10 y p menor a 0,01 |
| Distancia categórica | Mayor o igual a 0,10 |
| Diferencia de nulos | Al menos 10 puntos porcentuales |

Con menos de 20 registros utilizables se señala información insuficiente para la comparación, salvo que el cambio de nulos ya active una alerta.

Los umbrales deben calibrarse con el negocio. Una alerta de drift no demuestra pérdida de precisión ni activa reentrenamiento automático.

### Ejecución del monitoreo

Demostración con entrenamiento frente a prueba:

```powershell
.\.venv\Scripts\python.exe .\mlops_pipeline\model_monitoring.py
```

Genera `reporte_drift.csv`. Esta división aleatoria no representa evolución temporal.

Procesar un CSV una sola vez:

```powershell
.\.venv\Scripts\python.exe .\mlops_pipeline\model_monitoring.py --actual lote_actual.csv
```

Revisar el CSV cada 60 segundos:

```powershell
.\.venv\Scripts\python.exe .\mlops_pipeline\model_monitoring.py --actual lote_actual.csv --intervalo 60
```

El archivo indicado debe existir y contener las ocho entradas requeridas.

La revisión periódica funciona mientras el proceso permanezca abierto y el computador esté encendido. Se detiene con `Ctrl + C`.

### Persistencia e historial

Cada contenido nuevo se registra en `monitoreo.sqlite` con:

- Identificador y fecha UTC de procesamiento.
- Cantidad de registros y variables con alerta.
- Entradas del modelo y etiqueta opcional.
- Predicciones y probabilidades estimadas.
- Métricas de drift por variable.

La firma combina el contenido del CSV, el modelo y los datos de referencia. Si coinciden con un lote ya procesado, no se duplica.

Las fechas representan ejecuciones del monitoreo, no fechas de originación de los créditos. Se incorporó detección exploratoria de tendencias y cambios abruptos sobre los tres últimos lotes; las reglas se detallan en el reporte de resultados.

La base SQLite y los lotes locales están excluidos de Git.

### Comprobación funcional

Se creó un lote histórico de demostración con 2.153 registros del conjunto de prueba. No se detectaron alertas.

En una copia se multiplicó por diez `capital_prestado`. El monitoreo generó una alerta.

Esta comprobación demuestra la respuesta ante una modificación simulada. No constituye evidencia de drift real en producción.

### Aplicación Streamlit

La aplicación permite:

- Cargar archivos CSV y generar predicciones por lote.
- Descargar las predicciones y el reporte de drift.
- Comparar distribuciones históricas y actuales.
- Consultar indicadores y alertas.
- Evaluar las predicciones cuando existen etiquetas válidas.
- Consultar el historial automático de SQLite.

Ejecutar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Dirección habitual: http://localhost:8501

Sin un CSV cargado se utiliza el conjunto de prueba como demostración. En el historial, el botón de actualización vuelve a consultar los lotes guardados.

## Entrega final: API y Docker

### API de predicción por lotes

La API utiliza FastAPI y carga el pipeline local `mejor_modelo.joblib` al iniciar.

Ejecutar:

```powershell
.\.venv\Scripts\python.exe -m uvicorn model_deploy:app --app-dir mlops_pipeline --host 127.0.0.1 --port 8000
```

Documentación interactiva: http://127.0.0.1:8000/docs

| Endpoint | Función |
|---|---|
| `GET /health` | Comprobar disponibilidad del modelo |
| `POST /predict` | Predecir entre 1 y 10.000 registros enviados en JSON |

### Ejemplo de solicitud

```json
{
  "registros": [
    {
      "tipo_credito": 4,
      "capital_prestado": 2000000,
      "plazo_meses": 12,
      "edad_cliente": 35,
      "tipo_laboral": "Empleado",
      "salario_cliente": 3000000,
      "total_otros_prestamos": 500000,
      "cuota_pactada": 200000
    }
  ]
}
```

La respuesta incluye el número de registros, el índice de cada entrada, la clase predicha y `probabilidad_modelo_clase_0`.

Las probabilidades son estimaciones no calibradas. La respuesta conserva el orden de entrada.

La API aplica validaciones de positividad, edad, campos requeridos y tamaño del lote. Las solicitudes inválidas reciben código 422.

Solo se carga el modelo local del proyecto; no se aceptan archivos de modelos enviados por usuarios.

### Docker

La imagen utiliza Python 3.13 e instala `requirements-runtime.lock` con `--require-hashes` y `--only-binary=:all:`. El bloqueo generado contiene 53 paquetes. Esto fija versiones y verifica los archivos descargados; no garantiza por sí solo ausencia de vulnerabilidades.

Con Docker Desktop iniciado, construir:

```powershell
docker build -t creditos-ml:1.1 .
```

Crear e iniciar el contenedor:

```powershell
docker run -d --name creditos-api-prueba -p 127.0.0.1:8002:8000 creditos-ml:1.1
```

Documentación de la API: http://127.0.0.1:8002/docs

Si el contenedor ya existe y está detenido:

```powershell
docker start creditos-api-prueba
```

Consultar estado y registros:

```powershell
docker ps
docker logs creditos-api-prueba
```

Detener:

```powershell
docker stop creditos-api-prueba
```

La API se ejecuta con un usuario sin privilegios. El contenedor incluye una comprobación periódica de `/health`.

Se verificaron el estado `healthy`, la respuesta 200 de `/health` y la predicción de dos registros mediante `/predict`.

La imagen incluye las dependencias de Streamlit, pero su comando de inicio predeterminado ejecuta la API.

Después de modificar el código, la imagen debe reconstruirse y el contenedor debe recrearse para utilizar la nueva versión.

## Pruebas y cobertura

`test_proyecto.py` verifica:

- Disponibilidad de la API.
- Predicciones por lote y conservación del orden.
- Rechazo de entradas inválidas.
- Transformación de nulos y categorías desconocidas.
- Cálculo de PSI y detección de cambios simulados.
- Validación de columnas e información insuficiente.
- Persistencia de predicciones y prevención de duplicados.

Ejecutar:

```powershell
.\.venv\Scripts\python.exe -m pytest test_proyecto.py -v --cov --cov-report=term-missing --cov-report=xml:coverage.xml
```

La última ejecución local documentada aprobó **33 pruebas**, con una advertencia, en **21,41 segundos**. La cobertura total fue **76 %**. El detalle aparece en el reporte de resultados al final de este README.

Permanece un aviso de deprecación de Starlette/httpx que no impidió aprobar las pruebas. `coverage.xml` se genera al ejecutar la suite y no se guarda en Git.

## Análisis de calidad con SonarCloud

El workflow `.github/workflows/sonar.yml` se activa al publicar cambios en `developer`.

Sus pasos son:

1. Descargar el repositorio.
2. Preparar Python 3.13.
3. Instalar dependencias de ejecución y pruebas.
4. Ejecutar pytest y generar `coverage.xml`.
5. Enviar el código y la cobertura a SonarCloud.

Proyecto de Sonar:

https://sonarcloud.io/dashboard?id=xALEJOXx_Repositorio-base

Identificadores:

```properties
sonar.projectKey=xALEJOXx_Repositorio-base
sonar.organization=xalejoxx
```

El token se almacena como secreto `SONAR_TOKEN` en GitHub.

**Último resultado documentado:** Quality Gate aprobado, 83,01 % de cobertura del código nuevo, cero incidencias nuevas y 0 % de duplicación. Esa captura es anterior a la incorporación temporal; debe revisarse un análisis posterior para afirmar el estado de esos últimos cambios.

El workflow actual ejecuta el scanner, pero no espera explícitamente el resultado del Quality Gate. Una ejecución verde de Actions no sustituye la consulta del panel de SonarCloud.

## Limitaciones

- La interpretación de las clases, las unidades monetarias y la disponibilidad temporal de las entradas requieren confirmación.
- La división aleatoria no demuestra desempeño sobre créditos futuros.
- Sin un identificador confirmado no puede garantizarse separación de créditos del mismo cliente.
- El EDA utilizó la base completa; por ello, la prueba no es totalmente independiente de las decisiones exploratorias.
- Random Forest presenta desempeño limitado en la clase minoritaria y señales de sobreajuste.
- Las probabilidades no están calibradas.
- El monitoreo se comprobó con datos históricos y un cambio simulado, no con lotes nuevos de producción.
- La detección temporal utiliza tres ejecuciones y reglas exploratorias que requieren calibración.
- No se realizaron pruebas de carga o escalabilidad.
- La API aplica reglas de entrada más estrictas que la limpieza interna del pipeline.
- El último Quality Gate documentado debe actualizarse para incluir los cambios temporales posteriores.
- La revisión periódica requiere mantener el proceso y el equipo encendidos; no se configuró como servicio permanente.

El proyecto es una implementación académica y no constituye un sistema operativo de aprobación de créditos.

## Reporte de resultados y evidencias

### Valores principales del experimento

| Dato | Valor | Lectura del resultado |
|---|---:|---|
| Registros originales | 10.763 | Base utilizada para el ejercicio |
| Variables originales | 23 | Se seleccionaron ocho entradas para modelado |
| Clase 0 | 511, aproximadamente 4,75 % | Es la clase minoritaria |
| Clase 1 | 10.252, aproximadamente 95,25 % | Domina el conjunto de datos |
| Entrenamiento | 8.610 | Comparación con validación cruzada de cinco particiones |
| Prueba | 2.153 | Evaluación del modelo seleccionado |
| Semilla | 42 | Permite repetir la separación aleatoria |

Random Forest obtuvo el mayor F1 medio de clase 0 en validación, **0,1361**. Sin embargo, la brecha entre entrenamiento y validación fue **0,3622**, señal de sobreajuste. La selección es relativa a los modelos probados; no significa que el desempeño sea suficiente para el negocio.

En prueba detectó **29 de 102 casos de clase 0** y omitió 73. Además, generó 263 predicciones de clase 0 sobre registros de clase 1. El recall de clase 0 fue **28,43 %**, la precisión **9,93 %** y el F1 **0,1472**. La matriz y las métricas completas se incluyen en la sección de entrenamiento.

### Comprobación del monitoreo

| Comprobación realizada | Resultado observado | Alcance |
|---|---|---|
| Entrenamiento frente a 2.153 registros de prueba | Cero variables con alerta | Comparación histórica, sin separación temporal |
| Copia con `capital_prestado` multiplicado por diez | Una variable con alerta | Cambio simulado para verificar la respuesta |
| Repetir el mismo lote, modelo y referencia | No duplicar el registro | La firma combina CSV, modelo y referencia |
| Archivo fuera de la carpeta del proyecto o sin extensión CSV | Rechazo | Validación de la ruta antes de leer el lote |

El código sí incluye revisión periódica mediante `--intervalo 60`. Se ejecuta mientras el proceso permanezca abierto. Como mejora operativa queda instalarlo como servicio o tarea administrada.

Si hay alertas, el proceso recomienda revisar calidad, origen y población, y evaluar el desempeño con etiquetas antes de reentrenar. El sistema no reentrena automáticamente.

### Reglas de evolución temporal

La función `analizar_evolucion` ordena los lotes por `fecha_utc` y revisa los tres últimos valores de `variables_alerta`.

| Condición | Estado |
|---|---|
| Menos de tres lotes | `DATOS_INSUFICIENTES` |
| Diferencia absoluta de al menos dos variables entre los dos últimos lotes | `CAMBIO_ABRUPTO` |
| Dos incrementos consecutivos | `TENDENCIA_CRECIENTE` |
| Dos disminuciones consecutivas | `TENDENCIA_DECRECIENTE` |
| Ninguna de las anteriores | `SIN_TENDENCIA_DETECTADA` |

El cambio abrupto tiene prioridad. Por ejemplo, 0, 1, 2 indica crecimiento y 0, 1, 4 indica cambio abrupto. Son ejemplos de las reglas, no observaciones de producción. Las fechas corresponden al procesamiento de los lotes.

### Evidencia de pruebas locales

Salida final registrada: **33 passed, 1 warning in 21.41s**. Entorno reportado: Windows y Python 3.13.12.

| Componente | Sentencias | Sin ejecutar | Cobertura |
|---|---:|---:|---:|
| `app.py` | 144 | 22 | 85 % |
| `generar_lock.py` | 31 | 31 | 0 % |
| `mlops_pipeline/__init__.py` | 0 | 0 | 100 % |
| Ingeniería de características | 74 | 24 | 68 % |
| API | 46 | 0 | 100 % |
| Monitoreo | 194 | 41 | 79 % |
| Entrenamiento y evaluación | 98 | 21 | 79 % |
| **Total** | **587** | **139** | **76 %** |

Los porcentajes están redondeados como en pytest. Un archivo con cero sentencias no aporta validaciones adicionales. La cobertura mide código ejecutado y no garantiza por sí sola que todos los comportamientos estén verificados.

La suite se amplió para comprobar rutas externas, archivos no CSV, disponibilidad sin modelo, limpieza al cerrar o ante errores, evaluación y guardado temporal, estados del historial de Streamlit y reglas de evolución. Los datos sintéticos y las bases temporales de las pruebas no sustituyen el modelo ni el historial real.

### Evidencia del análisis estático

| Indicador | Último valor comprobado |
|---|---:|
| Quality Gate | Passed |
| Cobertura del código nuevo | 83,01 % |
| Mínimo exigido en ese análisis | 80 % |
| Incidencias nuevas | 0 |
| Duplicación del código nuevo | 0,0 % |

Estos valores corresponden a la captura anterior a los últimos cambios temporales. El **76 % local** es cobertura total; el **83,01 % de SonarCloud** es cobertura del código nuevo y corresponde a otro momento. No son la misma medición.

### Correcciones realizadas

| Problema observado | Motivo | Cambio aplicado |
|---|---|---|
| SonarCloud rechazó el análisis de CI | Automatic Analysis estaba habilitado al mismo tiempo | Se desactivó el análisis automático y se conservó GitHub Actions |
| Riesgo de lectura de rutas externas | El monitoreo recibía una ruta de archivo | Se resuelve la ruta y se exige ubicación dentro del proyecto y extensión CSV |
| Dependencias de Docker sin bloqueo completo | La instalación podía resolver versiones diferentes | Lock con versiones, hashes y paquetes binarios |
| Limpieza del modelo ante una excepción | El cierre debía liberar el recurso incluso con errores | Limpieza con `try/finally` y pruebas del ciclo de vida |
| Cobertura insuficiente | Faltaban casos de interfaz, entrenamiento y errores | Ampliación hasta 33 pruebas aprobadas |
| Fallo en un mensaje de Streamlit | Una tilde estaba mal codificada en la comprobación | Corrección del texto y repetición de las pruebas |

Las primeras ejecuciones de Actions quedaron en cola y no permitían cancelarse. La página de estado consultada mostraba una incidencia del servicio; después hubo ejecuciones satisfactorias. Este episodio no se atribuye al modelo.

### Dónde revisar y repetir las evidencias

| Evidencia | Ubicación |
|---|---|
| Carga, tipos de datos, nulos y gráficos | Notebooks de `mlops_pipeline` |
| Comparación de modelos | `comparacion_modelos.csv` y `comparacion_f1.png` |
| Desempeño final | `metricas_prueba.json` y `matriz_confusion_prueba.png` |
| Métricas de drift | `reporte_drift.csv`, Streamlit y SQLite local |
| Reglas temporales | `analizar_evolucion` en `model_monitoring.py` y sus pruebas |
| Validación automatizada | `test_proyecto.py` y registros de [GitHub Actions](https://github.com/xALEJOXx/Repositorio-base-/actions) |
| Calidad estática | [Panel de SonarCloud](https://sonarcloud.io/dashboard?id=xALEJOXx_Repositorio-base) |
| Estado del servicio | `/health`, Swagger y registros de Docker |

Repetir las pruebas desde PowerShell sin abrir ventanas de gráficos:

```powershell
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -m pytest test_proyecto.py -v --cov --cov-report=term-missing --cov-report=xml:coverage.xml
```

Comprobar el contenedor de prueba:

```powershell
docker ps --filter "name=creditos-api-prueba"
docker logs --tail 30 creditos-api-prueba
Invoke-RestMethod http://127.0.0.1:8002/health
```

El contenedor de prueba fue observado en estado `healthy` y sus registros mostraron `GET /health` con respuesta 200. Debe volver a comprobarse al iniciar la demostración.

## Balance de la entrega

Se implementó el recorrido desde la carga y exploración hasta la predicción por lotes, Streamlit, monitoreo, persistencia, Docker y controles automáticos de código. El monitoreo periódico y las recomendaciones ante alertas están implementados.

Jenkins no se implementó. En la guía corresponde al componente adicional; GitHub Actions no se presenta como una implementación de Jenkins. También debe comprobarse el acceso del docente `juanseparracourses` al repositorio según las instrucciones del taller.

La principal limitación es el desempeño de la clase minoritaria. Antes de utilizar el modelo en decisiones reales se requiere confirmar la definición del objetivo, revisar los datos, mejorar la detección y validar con información temporal y de negocio. Aprobar las pruebas de software y el Quality Gate no sustituye esa evaluación.
