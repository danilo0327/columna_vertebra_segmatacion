#!/bin/bash
# Script de despliegue automatizado en EC2 con Docker (Linux/Mac)
# Uso: ./scripts/deploy_ec2_docker.sh

set -e

# Configuración
EC2IP="${EC2_IP:-100.28.216.213}"
KEY_PATH="${EC2_KEY_PATH:-$HOME/Downloads/final_key.pem}"
KEY_NAME="${EC2_KEY_NAME:-final_key.pem}"
USER="${EC2_USER:-ubuntu}"

echo "========================================"
echo "Despliegue en EC2 con Docker"
echo "========================================"
echo ""

# Verificar que existe la key
if [ ! -f "$KEY_PATH" ]; then
    echo "ERROR: No se encuentra la key en: $KEY_PATH"
    exit 1
fi

# Ajustar permisos de la key
chmod 400 "$KEY_PATH" 2>/dev/null || true

echo "✓ Key encontrada: $KEY_PATH"

# Verificar conexión SSH
echo ""
echo "Verificando conexión SSH..."
if ! ssh -i "$KEY_PATH" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$USER@$EC2IP" "echo 'OK'" > /dev/null 2>&1; then
    echo "ERROR: No se puede conectar a EC2. Verifica:"
    echo "  - IP pública: $EC2IP"
    echo "  - Security Group permite SSH (puerto 22)"
    echo "  - La instancia está corriendo"
    exit 1
fi
echo "✓ Conexión SSH exitosa"

# Instalar Docker en EC2
echo ""
echo "========================================"
echo "Instalando Docker en EC2..."
echo "========================================"

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$USER@$EC2IP" << 'ENDSSH'
set -e

# Instalar Docker si no está instalado
if ! command -v docker &> /dev/null; then
    echo "Instalando Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo usermod -aG docker ubuntu
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "✓ Docker instalado"
else
    echo "✓ Docker ya está instalado"
fi

# Verificar que Docker funciona
sudo docker --version

# Preparar directorio
mkdir -p /home/ubuntu/columna_vertebra_segmatacion
cd /home/ubuntu/columna_vertebra_segmatacion

# Limpiar contenedores anteriores
if sudo docker ps -a | grep -q segmentacion-columna; then
    echo "Deteniendo y eliminando contenedor anterior..."
    sudo docker stop segmentacion-columna 2>/dev/null || true
    sudo docker rm segmentacion-columna 2>/dev/null || true
fi

# Limpiar imagen anterior
if sudo docker images | grep -q segmentacion-columna; then
    echo "Eliminando imagen anterior..."
    sudo docker rmi segmentacion-columna 2>/dev/null || true
fi
ENDSSH

# Crear .dockerignore si no existe
if [ ! -f .dockerignore ]; then
    cat > .dockerignore << 'EOF'
venv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
docs/
notebooks/
EstadoDeArte/
data/
*.log
.env
.DS_Store
Thumbs.db
EOF
    echo "✓ Creado .dockerignore"
fi

# Comprimir proyecto
echo ""
echo "========================================"
echo "Comprimiendo proyecto..."
echo "========================================"

ZIP_FILE="segmentacion-columna-deploy.zip"
rm -f "$ZIP_FILE"

zip -r "$ZIP_FILE" \
    segmentacion_app \
    models \
    scripts \
    Dockerfile \
    .dockerignore \
    run.py \
    -x "*.pyc" "*__pycache__*" "*.log" > /dev/null 2>&1

echo "✓ Proyecto comprimido: $ZIP_FILE"

# Subir archivo
echo ""
echo "========================================"
echo "Subiendo proyecto a EC2..."
echo "========================================"

scp -i "$KEY_PATH" -o StrictHostKeyChecking=no "$ZIP_FILE" "$USER@$EC2IP:/tmp/"
echo "✓ Archivo subido"

# Descomprimir y construir
echo ""
echo "========================================"
echo "Construyendo y ejecutando Docker..."
echo "========================================"

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$USER@$EC2IP" << ENDSSH
set -e

cd /home/ubuntu/columna_vertebra_segmatacion

# Descomprimir
echo "Descomprimiendo proyecto..."
unzip -o /tmp/$ZIP_FILE -d . > /dev/null 2>&1
rm -f /tmp/$ZIP_FILE

# Construir imagen Docker
echo ""
echo "Construyendo imagen Docker (esto puede tardar 10-15 minutos)..."
sudo docker build -t segmentacion-columna .

# Ejecutar contenedor
echo ""
echo "Ejecutando contenedor..."
sudo docker run -d \
  --name segmentacion-columna \
  -p 8000:8000 \
  --restart unless-stopped \
  segmentacion-columna

echo "✓ Contenedor ejecutándose"

# Esperar y verificar
sleep 5
echo ""
echo "Verificando estado..."
sudo docker ps | grep segmentacion-columna

echo ""
echo "Verificando salud de la aplicación..."
sleep 3
curl -s http://localhost:8000/api/health || echo "La aplicación aún está iniciando..."
ENDSSH

# Limpiar
rm -f "$ZIP_FILE"

echo ""
echo "========================================"
echo "✓ DESPLIEGUE COMPLETADO"
echo "========================================"
echo ""
echo "La aplicación está disponible en:"
echo "  http://$EC2IP:8000"
echo ""
echo "Para ver los logs:"
echo "  ssh -i $KEY_PATH $USER@$EC2IP"
echo "  sudo docker logs -f segmentacion-columna"
echo ""


