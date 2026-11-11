@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto instalar

echo Creando entorno virtual con Python 3.13...
py -3.13 -m venv .venv
if errorlevel 1 goto error

:instalar
echo Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo Verificando las librerias...
".venv\Scripts\python.exe" -c "import pandas, numpy, matplotlib, seaborn, openpyxl; print('Librerias verificadas correctamente')"
if errorlevel 1 goto error

echo.
echo Entorno preparado correctamente.
pause
exit /b 0

:error
echo.
echo No se pudo completar la preparacion. Revisa el error anterior.
pause
exit /b 1