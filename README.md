# Segmentación de Columna Vertebral

Aplicación web para segmentación automática de columna vertebral y vértebra T1 en radiografías usando DeepLabV3+.

## 🏗️ Estructura del Proyecto

```
columna_vertebra_segmatacion/
├── segmentacion_app/          # Aplicación principal
│   ├── app/
│   │   ├── api.py            # Endpoints de la API
│   │   ├── config.py         # Configuración
│   │   ├── main.py           # Aplicación principal
│   │   ├── model/            # Modelos ML
│   │   │   └── segmentation_model.py
│   │   ├── schemas/          # Esquemas Pydantic
│   │   │   └── segmentation.py
│   │   ├── static/           # Archivos estáticos
│   │   ├── templates/        # Templates HTML
│   │   │   └── index.html
│   │   └── tests/            # Tests
│   └── requirements.txt      # Dependencias
├── data/                     # Datos de entrenamiento
├── models/                   # Modelos entrenados
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Scripts de utilidad
│   ├── extract_model.py
│   └── run_server.py
├── classes_deeplabv3plus.json  # Clases del modelo
├── deeplabv3plus_20251114_040131.zip  # Modelo comprimido
├── Dockerfile                # Configuración Docker
└── README.md                 # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.10 o superior
- pip

### Pasos de Instalación

1. **Clonar o descargar el proyecto**

2. **Crear entorno virtual (OBLIGATORIO - evita conflictos de dependencias)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install --upgrade pip
pip install -r segmentacion_app/requirements.txt
```

4. **Verificar instalación**
```bash
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import fastapi; print('fastapi instalado')"
```

5. **Extraer el modelo (opcional, se extrae automáticamente al usar)**
```bash
python scripts/extract_model.py
```

## 💻 Uso Local

### Ejecutar el servidor de desarrollo

**Opción 1: Usando el script de inicio (Recomendado)**
```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh
./start.sh
```

**Opción 2: Manualmente**
```bash
# Activar entorno virtual primero
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

python run.py
```

**Opción 3: Directamente con uvicorn**
```bash
uvicorn segmentacion_app.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Acceder a la aplicación

Abre tu navegador en: `http://localhost:8000`

### Verificar que funciona

1. **Endpoint de salud:**
```bash
curl http://localhost:8000/api/health
```

Debería responder con:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "device": "cpu",
  "classes": ["Background", "T1", "V"]
}
```

2. **Interfaz web:**
   - Abre `http://localhost:8000` en tu navegador
   - Deberías ver la interfaz para cargar imágenes

## 🐳 Despliegue con Docker

### Construir la imagen

```bash
docker build -t segmentacion-columna .
```

### Ejecutar el contenedor

```bash
docker run -p 8000:8000 segmentacion-columna
```

## ☁️ Despliegue en AWS EC2

### Opción 1: Sin Docker

1. **Conectar a tu instancia EC2**
```bash
ssh -i tu-key.pem ubuntu@tu-ec2-ip
```

2. **Instalar dependencias del sistema**
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git
```

3. **Clonar o subir el proyecto**
```bash
# Subir archivos usando SCP o clonar desde Git
```

4. **Configurar la aplicación**
```bash
cd columna_vertebra_segmatacion
python3 -m venv venv
source venv/bin/activate
pip install -r segmentacion_app/requirements.txt
```

5. **Ejecutar con systemd (recomendado)**

Crear archivo `/etc/systemd/system/segmentacion.service`:

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

[Install]
WantedBy=multi-user.target
```

Activar el servicio:
```bash
sudo systemctl daemon-reload
sudo systemctl enable segmentacion
sudo systemctl start segmentacion
sudo systemctl status segmentacion
```

6. **Configurar seguridad (Security Groups)**

Asegúrate de que el Security Group de tu EC2 permita tráfico HTTP/HTTPS en el puerto 8000 (o el que uses).

### Opción 2: Con Docker

1. **Instalar Docker en EC2**
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

2. **Construir y ejecutar**
```bash
docker build -t segmentacion-columna .
docker run -d -p 8000:8000 --name segmentacion segmentacion-columna
```

## 📡 API Endpoints

### `GET /`
Interfaz web principal

### `POST /api/segment`
Segmenta una imagen de radiografía

**Parámetros:**
- `file`: Archivo de imagen (PNG, JPG, DICOM)

**Respuesta:**
```json
{
  "success": true,
  "message": "Segmentación completada exitosamente",
  "original_image_url": "/static/original_xxx.png",
  "segmented_image_url": "/static/mask_xxx.png",
  "overlay_image_url": "/static/overlay_xxx.png",
  "classes_detected": ["Background", "T1", "V"]
}
```

### `GET /api/health`
Verifica el estado de la API

## 🔧 Configuración

Las configuraciones principales están en `segmentacion_app/app/config.py`:

- `INPUT_SIZE`: Tamaño de entrada del modelo (512, 512)
- `NUM_CLASSES`: Número de clases (3: Background, T1, V)
- `MAX_FILE_SIZE`: Tamaño máximo de archivo (10MB)
- `ALLOWED_EXTENSIONS`: Extensiones permitidas

## 📝 Notas

- El modelo se extrae automáticamente del ZIP la primera vez que se usa
- Las imágenes procesadas se guardan en `segmentacion_app/app/static/`
- El modelo se carga en GPU si está disponible, sino usa CPU

## 🐛 Solución de Problemas

### Error al cargar el modelo
- Verifica que el archivo ZIP existe y está en la raíz del proyecto
- Verifica que el archivo JSON de clases existe
- Revisa los logs para más detalles

### Error de memoria
- Reduce el tamaño de entrada en `config.py`
- Usa una instancia EC2 con más RAM

### Puerto ya en uso
- Cambia el puerto en `config.py` o usa la variable de entorno `PORT`

## 📄 Licencia

Este proyecto es para uso académico/investigación.

