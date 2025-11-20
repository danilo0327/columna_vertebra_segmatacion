# Script para iniciar el servidor de segmentación
Write-Host "Iniciando servidor de segmentación..." -ForegroundColor Green

# Activar entorno virtual
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    Write-Host "Entorno virtual activado" -ForegroundColor Green
} else {
    Write-Host "ERROR: Entorno virtual no encontrado. Ejecuta: python -m venv venv" -ForegroundColor Red
    exit 1
}

# Cambiar al directorio de la aplicación
Set-Location segmentacion_app

# Iniciar servidor
Write-Host "`nServidor iniciándose en http://localhost:8000" -ForegroundColor Cyan
Write-Host "Presiona Ctrl+C para detener el servidor`n" -ForegroundColor Yellow

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

