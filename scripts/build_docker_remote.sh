#!/bin/bash
set -e
cd /home/ubuntu/columna_vertebra_segmatacion
echo "Descomprimiendo proyecto..."
unzip -o /tmp/segmentacion-columna-deploy.zip -d . > /dev/null 2>&1
rm -f /tmp/segmentacion-columna-deploy.zip

# Si se clonó desde git, descargar modelos con Git LFS
if [ -d .git ]; then
    echo ""
    echo "Descargando modelos con Git LFS..."
    if command -v git-lfs &> /dev/null; then
        git lfs install
        git lfs pull
        echo "✓ Modelos descargados"
    else
        echo "⚠️  Git LFS no está instalado. Instalando..."
        sudo apt-get update
        sudo apt-get install -y git-lfs
        git lfs install
        git lfs pull
        echo "✓ Modelos descargados"
    fi
fi

echo ""
echo "Construyendo imagen Docker (esto puede tardar 10-15 minutos)..."
sudo docker build -t segmentacion-columna .
echo ""
echo "Ejecutando contenedor..."
sudo docker run -d --name segmentacion-columna -p 8000:8000 --restart unless-stopped segmentacion-columna
echo "✓ Contenedor ejecutándose"
sleep 5
echo ""
echo "Verificando estado..."
sudo docker ps | grep segmentacion-columna
echo ""
echo "Verificando salud de la aplicación..."
sleep 3
curl -s http://localhost:8000/api/health || echo "La aplicacion aun esta iniciando..."

