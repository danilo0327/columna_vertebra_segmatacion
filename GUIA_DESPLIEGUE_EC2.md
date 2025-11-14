# Guía de Despliegue en AWS EC2

Esta guía te ayudará a desplegar la aplicación de segmentación en una instancia EC2 de AWS.

## 📋 Requisitos Previos

- Una instancia EC2 corriendo (Ubuntu recomendado)
- Acceso SSH a la instancia
- Security Group configurado para permitir tráfico en el puerto 8000 (o el que uses)

## 🚀 Opción 1: Subir Archivos Directamente (Más Rápido)

### Paso 1: Preparar los archivos localmente

1. **Comprimir el proyecto** (excluyendo venv y archivos innecesarios):

   **En Windows PowerShell:**
   ```powershell
   Compress-Archive -Path . -DestinationPath segmentacion-columna.zip -Exclude "venv","__pycache__",".git","*.pyc"
   ```

   **En Linux/Mac:**
   ```bash
   zip -r segmentacion-columna.zip . -x "venv/*" "__pycache__/*" ".git/*" "*.pyc"
   ```

### Paso 2: Subir a EC2

```bash
scp -i tu-key.pem segmentacion-columna.zip ubuntu@tu-ec2-ip:/home/ubuntu/
```

### Paso 3: Conectar a EC2 y configurar

```bash
# Conectar a EC2
ssh -i tu-key.pem ubuntu@tu-ec2-ip

# Descomprimir
cd /home/ubuntu
unzip segmentacion-columna.zip -d columna_vertebra_segmatacion
cd columna_vertebra_segmatacion

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r segmentacion_app/requirements.txt

# Probar que funciona
python3 run.py
```

Presiona `Ctrl+C` para detener el servidor de prueba.

### Paso 4: Configurar como servicio systemd (Recomendado)

Crea el archivo de servicio:

```bash
sudo nano /etc/systemd/system/segmentacion.service
```

Pega el siguiente contenido (ajusta las rutas según tu caso):

```ini
[Unit]
Description=Segmentacion Columna Vertebral API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/columna_vertebra_segmatacion
Environment="PATH=/home/ubuntu/columna_vertebra_segmatacion/venv/bin"
ExecStart=/home/ubuntu/columna_vertebra_segmatacion/venv/bin/uvicorn segmentacion_app.app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Guarda y cierra (Ctrl+X, Y, Enter).

### Paso 5: Activar el servicio

```bash
# Recargar systemd
sudo systemctl daemon-reload

# Habilitar el servicio (inicia automáticamente al reiniciar)
sudo systemctl enable segmentacion

# Iniciar el servicio
sudo systemctl start segmentacion

# Verificar estado
sudo systemctl status segmentacion
```

### Paso 6: Verificar que funciona

```bash
# Ver logs en tiempo real
sudo journalctl -u segmentacion -f

# Probar el endpoint
curl http://localhost:8000/api/health
```

## 🐳 Opción 2: Usar Docker

### Paso 1: Instalar Docker en EC2

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
# Cerrar sesión y volver a conectar para que los cambios surtan efecto
```

### Paso 2: Subir archivos (igual que Opción 1)

### Paso 3: Construir y ejecutar

```bash
cd columna_vertebra_segmatacion
docker build -t segmentacion-columna .
docker run -d -p 8000:8000 --name segmentacion --restart unless-stopped segmentacion-columna
```

### Paso 4: Verificar

```bash
# Ver logs
docker logs -f segmentacion

# Probar
curl http://localhost:8000/api/health
```

## 🔧 Configuración del Security Group

1. Ve a la consola de AWS EC2
2. Selecciona tu instancia
3. Ve a "Security Groups"
4. Edita las reglas de entrada (Inbound rules)
5. Agrega una regla:
   - **Type:** Custom TCP
   - **Port:** 8000
   - **Source:** 0.0.0.0/0 (o tu IP específica para mayor seguridad)
   - **Description:** Segmentacion API

## 🌐 Acceder desde Internet

Una vez configurado, puedes acceder a tu aplicación desde:
```
http://tu-ec2-ip-publica:8000
```

O si configuraste un dominio:
```
http://tu-dominio.com:8000
```

## 📝 Comandos Útiles

### Gestionar el servicio systemd

```bash
# Ver estado
sudo systemctl status segmentacion

# Iniciar
sudo systemctl start segmentacion

# Detener
sudo systemctl stop segmentacion

# Reiniciar
sudo systemctl restart segmentacion

# Ver logs
sudo journalctl -u segmentacion -f

# Ver últimas 100 líneas
sudo journalctl -u segmentacion -n 100
```

### Gestionar Docker

```bash
# Ver logs
docker logs -f segmentacion

# Detener
docker stop segmentacion

# Iniciar
docker start segmentacion

# Reiniciar
docker restart segmentacion

# Ver procesos
docker ps
```

## 🔍 Solución de Problemas

### El servicio no inicia

```bash
# Ver logs detallados
sudo journalctl -u segmentacion -n 50

# Verificar que el archivo de servicio está correcto
sudo systemctl cat segmentacion

# Verificar permisos
ls -la /home/ubuntu/columna_vertebra_segmatacion
```

### Error de permisos

```bash
# Dar permisos al usuario
sudo chown -R ubuntu:ubuntu /home/ubuntu/columna_vertebra_segmatacion
```

### Puerto ya en uso

```bash
# Ver qué está usando el puerto 8000
sudo lsof -i :8000

# O cambiar el puerto en config.py
```

### No puedo acceder desde Internet

1. Verifica el Security Group
2. Verifica que el servicio está corriendo: `sudo systemctl status segmentacion`
3. Verifica que está escuchando: `sudo netstat -tlnp | grep 8000`
4. Prueba desde la misma instancia: `curl http://localhost:8000/api/health`

## 🔄 Actualizar la Aplicación

### Si usas systemd:

```bash
cd /home/ubuntu/columna_vertebra_segmatacion
# Hacer pull de cambios o subir nuevos archivos
source venv/bin/activate
pip install -r segmentacion_app/requirements.txt
sudo systemctl restart segmentacion
```

### Si usas Docker:

```bash
cd /home/ubuntu/columna_vertebra_segmatacion
docker stop segmentacion
docker rm segmentacion
docker build -t segmentacion-columna .
docker run -d -p 8000:8000 --name segmentacion --restart unless-stopped segmentacion-columna
```

## 📊 Monitoreo

### Ver uso de recursos

```bash
# CPU y memoria
htop

# Espacio en disco
df -h

# Procesos
ps aux | grep uvicorn
```

### Logs de la aplicación

Los logs se guardan automáticamente por systemd. Para verlos:
```bash
sudo journalctl -u segmentacion -f
```

