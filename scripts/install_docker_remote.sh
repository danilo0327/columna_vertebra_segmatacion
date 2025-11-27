#!/bin/bash
set -e
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
sudo docker --version
mkdir -p /home/ubuntu/columna_vertebra_segmatacion
cd /home/ubuntu/columna_vertebra_segmatacion
if sudo docker ps -a | grep -q segmentacion-columna; then
    echo "Limpiando contenedor anterior..."
    sudo docker stop segmentacion-columna 2>/dev/null || true
    sudo docker rm segmentacion-columna 2>/dev/null || true
fi
if sudo docker images | grep -q segmentacion-columna; then
    echo "Limpiando imagen anterior..."
    sudo docker rmi segmentacion-columna 2>/dev/null || true
fi


