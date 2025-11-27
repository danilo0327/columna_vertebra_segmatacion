# Script de despliegue automatizado en EC2 con Docker
# Uso: .\scripts\deploy_ec2_docker.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$EC2IP = "100.28.216.213",
    
    [Parameter(Mandatory=$false)]
    [string]$KeyPath = "C:\Users\ASUS\Downloads\final_key.pem",
    
    [Parameter(Mandatory=$false)]
    [string]$User = "ubuntu"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Despliegue en EC2 con Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que existe la key
if (-not (Test-Path $KeyPath)) {
    Write-Host "ERROR: No se encuentra la key en: $KeyPath" -ForegroundColor Red
    exit 1
}

Write-Host " Key encontrada: $KeyPath" -ForegroundColor Green

# Verificar conexin SSH
Write-Host ""
Write-Host "Verificando conexin SSH..." -ForegroundColor Yellow
$null = ssh -i $KeyPath -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$User@$EC2IP" "echo OK" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: No se puede conectar a EC2. Verifica:" -ForegroundColor Red
    Write-Host "  - IP pblica: $EC2IP" -ForegroundColor Yellow
    Write-Host "  - Security Group permite SSH (puerto 22)" -ForegroundColor Yellow
    Write-Host "  - La instancia est corriendo" -ForegroundColor Yellow
    exit 1
}
Write-Host " Conexin SSH exitosa" -ForegroundColor Green

# Instalar Docker en EC2
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Paso 1: Instalando Docker en EC2..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Usar script bash predefinido para instalacin
$installScriptPath = Join-Path $PSScriptRoot "install_docker_remote.sh"
if (-not (Test-Path $installScriptPath)) {
    Write-Host "ERROR: No se encuentra install_docker_remote.sh" -ForegroundColor Red
    exit 1
}

$installFile = $installScriptPath

scp -i $KeyPath -o StrictHostKeyChecking=no $installFile "$User@${EC2IP}:/tmp/install_docker.sh"
ssh -i $KeyPath -o StrictHostKeyChecking=no "$User@$EC2IP" "chmod +x /tmp/install_docker.sh; bash /tmp/install_docker.sh"

# Crear .dockerignore
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Paso 2: Preparando archivos..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path .dockerignore)) {
    $ignoreLines = @("venv/", "__pycache__/", "*.pyc", ".git/", "*.md", "docs/", "notebooks/", "EstadoDeArte/", "data/", "*.log", ".env")
    $ignoreLines | Out-File -FilePath .dockerignore -Encoding UTF8
    Write-Host "Creado .dockerignore" -ForegroundColor Green
}

# Comprimir proyecto
Write-Host ""
Write-Host "Comprimiendo proyecto..." -ForegroundColor Yellow
$zipFile = "segmentacion-columna-deploy.zip"
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
}

Compress-Archive -Path segmentacion_app,models,scripts,Dockerfile,.dockerignore,run.py -DestinationPath $zipFile -Force
Write-Host " Proyecto comprimido" -ForegroundColor Green

# Subir archivo
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Paso 3: Subiendo proyecto a EC2..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Subiendo archivo (esto puede tardar varios minutos)..." -ForegroundColor Yellow
scp -i $KeyPath -o StrictHostKeyChecking=no $zipFile "$User@${EC2IP}:/tmp/"
Write-Host " Archivo subido" -ForegroundColor Green

# Construir y ejecutar
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Paso 4: Construyendo y ejecutando Docker..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Usar script bash predefinido
$buildScriptPath = Join-Path $PSScriptRoot "build_docker_remote.sh"
if (-not (Test-Path $buildScriptPath)) {
    Write-Host "ERROR: No se encuentra build_docker_remote.sh" -ForegroundColor Red
    exit 1
}

scp -i $KeyPath -o StrictHostKeyChecking=no $buildScriptPath "$User@${EC2IP}:/tmp/build_docker.sh"
ssh -i $KeyPath -o StrictHostKeyChecking=no "$User@$EC2IP" "chmod +x /tmp/build_docker.sh; bash /tmp/build_docker.sh"

# Limpiar
Remove-Item $zipFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " DESPLIEGUE COMPLETADO" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "La aplicacin est disponible en:" -ForegroundColor Cyan
Write-Host "  http://$EC2IP:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para ver los logs:" -ForegroundColor Cyan
Write-Host "  ssh -i $KeyPath $User@$EC2IP" -ForegroundColor Yellow
Write-Host "  sudo docker logs -f segmentacion-columna" -ForegroundColor Yellow
Write-Host ""


