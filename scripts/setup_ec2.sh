#!/bin/bash
# Script para configurar la aplicación en EC2

echo "=========================================="
echo "Configuración de Segmentación en EC2"
echo "=========================================="

# Actualizar sistema
echo "Actualizando sistema..."
sudo apt update
sudo apt upgrade -y

# Instalar dependencias del sistema
echo "Instalando dependencias del sistema..."
sudo apt install -y python3-pip python3-venv git nginx

# Crear directorio de la aplicación
APP_DIR="/home/ubuntu/segmentacion-columna"
echo "Creando directorio de aplicación en $APP_DIR..."
mkdir -p $APP_DIR
cd $APP_DIR

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias Python
echo "Instalando dependencias Python..."
pip install --upgrade pip
pip install -r segmentacion_app/requirements.txt

# Crear archivo de servicio systemd
echo "Creando servicio systemd..."
sudo tee /etc/systemd/system/segmentacion.service > /dev/null <<EOF
[Unit]
Description=Segmentacion Columna Vertebral API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/uvicorn segmentacion_app.app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Recargar systemd y habilitar servicio
echo "Configurando servicio..."
sudo systemctl daemon-reload
sudo systemctl enable segmentacion
sudo systemctl start segmentacion

# Verificar estado
echo "Verificando estado del servicio..."
sudo systemctl status segmentacion --no-pager

echo "=========================================="
echo "Configuración completada!"
echo "=========================================="
echo "La aplicación debería estar corriendo en http://tu-ec2-ip:8000"
echo ""
echo "Comandos útiles:"
echo "  Ver logs: sudo journalctl -u segmentacion -f"
echo "  Reiniciar: sudo systemctl restart segmentacion"
echo "  Detener: sudo systemctl stop segmentacion"

