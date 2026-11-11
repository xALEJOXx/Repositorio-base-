FROM python:3.13-slim

# Configuración de Python y ubicación de los módulos del modelo.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/mlops_pipeline \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Instalar las dependencias de ejecución.
COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

# Copiar el código, los datos de referencia y el modelo.
COPY mlops_pipeline/ ./mlops_pipeline/
COPY app.py Base_de_datos.csv mejor_modelo.joblib ./

# Ejecutar el servicio con un usuario sin privilegios.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Comprobar que la API responde y tiene el modelo cargado.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]