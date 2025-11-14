@echo off
REM Script para iniciar la aplicación en Windows
echo Activando entorno virtual...
call venv\Scripts\activate.bat
echo Iniciando servidor...
python run.py
pause

