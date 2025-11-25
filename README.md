# Segmentación de Columna Vertebral y Vértebra T1

Aplicación web para segmentación automática de columna vertebral (V) y vértebra T1 en radiografías usando modelos de Deep Learning (DeepLabV3+, U-Net++, DeepLabV3++ con Decoder Denso).

## 🎯 Características

- **Múltiples modelos disponibles:**
  - DeepLabV3++ (Decoder Denso) - Modelo híbrido con decoder denso
  - U-Net++ v2 - Arquitectura U-Net++ optimizada
  - DeepLabV3+ ResNet50 - Modelo estándar de torchvision

- **Segmentación de múltiples clases:**
  - F (Fondo/Background)
  - V (Columna vertebral) - Visualizada en verde
  - T1 (Vértebra T1) - Visualizada en rojo

- **Métricas de evaluación:**
  - IoU (Intersection over Union) por clase
  - Dice Score por clase
  - Confianza promedio
  - Porcentaje de cobertura por clase

- **Interfaz web intuitiva:**
  - Carga de imágenes (PNG, JPG, DICOM)
  - Visualización de resultados con superposición
  - Selección de modelo desde la interfaz
  - Botón para limpiar y cargar nueva imagen

## 🧠 Arquitectura del Modelo DeepLabV3+ ResNet50

El modelo principal utilizado es **DeepLabV3+ con backbone ResNet50**, una arquitectura de segmentación semántica de última generación que combina un encoder profundo con un decoder refinado.

### Estructura de la Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        ENCODER                                │
│  ┌──────────────┐                                            │
│  │ Input Image  │ → ResNet-50 Backbone                        │
│  │ (512×256×3)  │                                            │
│  └──────────────┘                                            │
│         │                                                     │
│         ├─→ L_e: High-level features (conv4_block6_2_relu)   │
│         │   └─→ Atrous Spatial Pyramid Pooling (ASPP)       │
│         │       ├─ 1×1 Convolution                           │
│         │       ├─ 3×3 Convolution (rate=6)                  │
│         │       ├─ 3×3 Convolution (rate=12)                 │
│         │       ├─ 3×3 Convolution (rate=18)                  │
│         │       ├─ Image Pooling                              │
│         │       └─ Concatenation → 1×1 Conv (ASPP Output)    │
│         │                                                      │
│         └─→ L_d: Low-level features (conv2_block3_2_relu)     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        DECODER                               │
│  ┌──────────────────┐                                       │
│  │ ASPP Output      │ → Upsample by 4                       │
│  └──────────────────┘                                       │
│         │                                                     │
│         │                                                     │
│  ┌──────────────────┐                                       │
│  │ L_d Features     │ → 1×1 Conv                            │
│  └──────────────────┘                                       │
│         │                                                     │
│         └─→ Concatenation                                    │
│             └─→ 3×3 Convolution                              │
│                 └─→ Upsample by 4                            │
│                     └─→ Segmentation Mask (512×256×3)         │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Principales

#### 1. **Encoder: ResNet-50 Backbone**
- **Función:** Extracción de características multiescala
- **Salidas:**
  - **L_e (High-level):** Características de alto nivel desde `conv4_block6_2_relu`
  - **L_d (Low-level):** Características de bajo nivel desde `conv2_block3_2_relu`

#### 2. **Atrous Spatial Pyramid Pooling (ASPP)**
- **Propósito:** Capturar contexto a múltiples escalas usando convoluciones atrous (dilated)
- **Componentes:**
  - 1×1 Convolución estándar
  - 3×3 Convoluciones atrous con tasas 6, 12 y 18
  - Image Pooling (Adaptive Average Pooling)
  - Concatenación y proyección final con 1×1 convolución

#### 3. **Decoder**
- **Función:** Refinamiento de la segmentación usando características de bajo nivel
- **Proceso:**
  1. Upsampling del output de ASPP (×4)
  2. Procesamiento de características de bajo nivel (L_d) con 1×1 convolución
  3. Concatenación de características de alto y bajo nivel
  4. Refinamiento con 3×3 convolución
  5. Upsampling final (×4) para obtener la máscara de segmentación

### Ventajas de esta Arquitectura

- **Contexto multiescala:** ASPP captura información a diferentes escalas espaciales
- **Refinamiento preciso:** El decoder combina características de alto y bajo nivel para bordes más precisos
- **Eficiencia:** ResNet-50 proporciona un buen balance entre precisión y velocidad

## 🎓 Configuración del Entrenamiento

El modelo DeepLabV3+ ResNet50 fue entrenado con la siguiente configuración:

### Dataset

- **Total de imágenes:** 174 radiografías válidas
- **Anotaciones:** 499 anotaciones en formato COCO
- **Split:**
  - **Train:** 70% (121 imágenes)
  - **Validation:** 15% (26 imágenes)
  - **Test:** 15% (27 imágenes)
- **Tamaño de imagen:** 512×256 píxeles
- **Clases:** 3 clases (F=Fondo, V=Columna, T1=Vértebra T1)

### Preprocesamiento

- **Resize:** Todas las imágenes se redimensionan a 512×256
- **Normalización:** Valores de píxel normalizados a [0, 1]
- **Data Augmentation:**
  - Random horizontal flip (50% probabilidad)
  - Interpolación: `INTER_AREA` para imágenes, `INTER_NEAREST` para máscaras

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| **Batch Size** | 4 |
| **Epochs** | 50 |
| **Learning Rate** | 3×10⁻⁴ (0.0003) |
| **Optimizer** | AdamW |
| **Weight Decay** | 1×10⁻⁴ |
| **Scheduler** | CosineAnnealingLR (T_max=50) |
| **Loss Function** | Combined Loss (CE + Dice) |
|   - CE Weight | 0.6 |
|   - Dice Weight | 0.4 |
| **Class Weights** | [0.05, 1.0, 3.0] (F, V, T1) |

### Función de Pérdida

Se utiliza una **pérdida combinada** que combina Cross-Entropy y Dice Loss:

```python
Loss = 0.6 × CrossEntropy + 0.4 × DiceLoss
```

- **Cross-Entropy:** Penaliza errores de clasificación
- **Dice Loss:** Enfocado en la superposición de regiones (útil para clases desbalanceadas)
- **Class Weights:** Pesos ajustados para manejar el desbalance (F >> V > T1)

### Métricas de Evaluación

- **IoU (Intersection over Union)** por clase
- **mIoU (mean IoU)** excluyendo fondo
- **Modelo guardado:** Se guarda el modelo con mejor IoU de T1 en validación

### Resultados del Entrenamiento

El modelo alcanzó los siguientes resultados en validación (mejor época):

- **mIoU (sin fondo):** ~0.66
- **IoU por clase:**
  - F (Fondo): ~0.97
  - V (Columna): ~0.65
  - T1 (Vértebra): ~0.66

## 📊 Ejemplo de Inferencia

A continuación se muestra un ejemplo de los resultados obtenidos con el modelo DeepLabV3+ ResNet50:

### Resultados Visuales

El modelo genera tres visualizaciones:

1. **Imagen Original:** La radiografía de entrada en escala de grises
2. **Máscara de Segmentación:** La máscara binaria con las clases segmentadas
   - Fondo en negro
   - Columna vertebral (V) en gris oscuro
   - Vértebra T1 en gris claro
3. **Superposición:** Combinación de la imagen original con la segmentación
   - **Columna vertebral (V):** Resaltada en **verde**
   - **Vértebra T1:** Resaltada en **rojo**

### Métricas de Ejemplo

Para una radiografía típica, el modelo genera las siguientes métricas:

#### Métricas Globales
- **IoU Promedio (Estimado):** 0.8785
- **Dice Promedio (Estimado):** 0.9330
- **Cobertura Foreground:** 9.98%
- **Clases Detectadas:** 3

#### Métricas por Clase

**V (Columna Vertebral):**
- Porcentaje: 9.54%
- IoU: 0.9669
- Dice: 0.9832
- Confianza: 0.9522

**T1 (Vértebra T1):**
- Porcentaje: 0.44%
- IoU: 0.7901
- Dice: 0.8828
- Confianza: 0.7462

**F (Fondo):**
- Porcentaje: 90.02%
- IoU: 0.9975
- Dice: 0.9987
- Confianza: 0.9885

#### Promedio (mean)
- IoU: 0.8785
- Dice: 0.9330

### Interpretación

- **Alto IoU y Dice para V:** El modelo segmenta la columna vertebral con alta precisión (IoU > 0.96)
- **Buen rendimiento en T1:** A pesar de ser una clase minoritaria, el modelo logra un IoU de ~0.79 para T1
- **Fondo bien identificado:** El fondo se segmenta casi perfectamente (IoU > 0.99)
- **Cobertura realista:** El 9.98% de cobertura foreground refleja la proporción real de la columna y T1 en las radiografías

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
│   │   ├── static/           # Archivos estáticos (imágenes procesadas)
│   │   ├── templates/        # Templates HTML
│   │   │   └── index.html
│   │   └── tests/            # Tests
│   └── requirements.txt      # Dependencias
├── models/                   # Modelos entrenados
│   ├── deeplab_densedecoder/ # DeepLabV3++ (Decoder Denso)
│   ├── unetplusplus_v2/     # U-Net++ v2
│   └── deeplab_resnet50/    # DeepLabV3+ ResNet50
├── notebooks/               # Jupyter notebooks de entrenamiento
├── scripts/                 # Scripts de utilidad
│   ├── diagnosticos/       # Scripts de diagnóstico
│   │   ├── analyze_deeplab_hybrid.py
│   │   ├── diagnostico_t1.py
│   │   └── diagnostico_t1_dense_decoder.py
│   ├── tests/              # Scripts de prueba
│   │   ├── test_classes.py
│   │   ├── test_improvements.py
│   │   ├── test_metrics_calculation.py
│   │   ├── test_model_loading.py
│   │   ├── test_new_model.py
│   │   └── test_t1_improvement.py
│   ├── extract_model.py
│   ├── inspect_model.py
│   ├── run_server.py
│   └── setup_ec2.sh
├── Dockerfile               # Configuración Docker
├── docker-compose.yml       # Orquestación Docker (opcional)
├── install_dependencies.bat # Instalación Windows
├── install_dependencies.sh  # Instalación Linux/Mac
├── start.bat               # Inicio Windows
├── start.sh                 # Inicio Linux/Mac
├── iniciar_servidor.ps1     # Inicio PowerShell
├── docs/                   # Documentación y guías
│   ├── GUIA_DESPLIEGUE_EC2.md
│   ├── INSTALACION.md
│   ├── NOTAS_MODELO.md
│   └── SOLUCION_PROBLEMAS_UNET.md
└── README.md               # Este archivo
```

## 📋 Requisitos Previos

- **Python:** 3.10 o superior
- **Sistema Operativo:** Windows, Linux o macOS
- **RAM:** Mínimo 4GB (recomendado 8GB+)
- **Espacio en disco:** ~2GB para modelos y dependencias
- **Git LFS:** Requerido para descargar modelos grandes (si usas Git)

## 🚀 Instalación Local

### Paso 1: Clonar el Repositorio

```bash
git clone <tu-repositorio-url>
cd columna_vertebra_segmatacion

# Si usas Git LFS (para modelos grandes)
git lfs install
git lfs pull
```

### Paso 2: Crear Entorno Virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

**Opción A: Script Automático (Recomendado)**

**Windows:**
```cmd
install_dependencies.bat
```

**Linux/Mac:**
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Opción B: Manual**

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar otras dependencias
pip install -r segmentacion_app/requirements.txt
```

**Nota:** Si tienes GPU NVIDIA con CUDA, instala PyTorch con soporte GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Paso 4: Verificar Instalación

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import fastapi; print('FastAPI instalado')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## 💻 Uso Local

### Opción 1: Script de Inicio (Recomendado)

**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**PowerShell:**
```powershell
.\iniciar_servidor.ps1
```

### Opción 2: Manualmente

```bash
# Activar entorno virtual primero
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Ejecutar servidor
uvicorn segmentacion_app.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Acceder a la Aplicación

Abre tu navegador en: **http://localhost:8000**

### Verificar que Funciona

**Endpoint de salud:**
```bash
curl http://localhost:8000/api/health
```

Debería responder:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "device": "cpu",
  "classes": ["F", "V", "T1"]
}
```

## 🐳 Despliegue con Docker

### Requisitos

- Docker instalado
- Docker Compose (opcional, para orquestación)

### Opción 1: Docker Simple

**1. Construir la imagen:**
```bash
docker build -t segmentacion-columna .
```

**2. Ejecutar el contenedor:**
```bash
docker run -d \
  --name segmentacion \
  -p 8000:8000 \
  --restart unless-stopped \
  segmentacion-columna
```

**3. Verificar:**
```bash
docker logs -f segmentacion
curl http://localhost:8000/api/health
```

### Opción 2: Docker Compose

**1. Crear `docker-compose.yml`:**
```yaml
version: '3.8'

services:
  segmentacion:
    build: .
    container_name: segmentacion-columna
    ports:
      - "8000:8000"
    restart: unless-stopped
    volumes:
      - ./models:/app/models
      - ./segmentacion_app/app/static:/app/segmentacion_app/app/static
    environment:
      - PORT=8000
      - HOST=0.0.0.0
```

**2. Ejecutar:**
```bash
docker-compose up -d
```

**3. Ver logs:**
```bash
docker-compose logs -f
```

### Comandos Útiles Docker

```bash
# Ver logs
docker logs -f segmentacion

# Detener
docker stop segmentacion

# Iniciar
docker start segmentacion

# Reiniciar
docker restart segmentacion

# Eliminar contenedor
docker rm segmentacion

# Eliminar imagen
docker rmi segmentacion-columna
```

## ☁️ Despliegue en AWS EC2

### Requisitos Previos

- Instancia EC2 corriendo (Ubuntu 20.04+ recomendado)
- Acceso SSH a la instancia
- Security Group configurado para permitir tráfico en puerto 8000
- Tipo de instancia: t3.medium o superior (recomendado para modelos grandes)

### Opción 1: Despliegue Directo (Sin Docker)

#### Paso 1: Conectar a EC2

```bash
ssh -i tu-key.pem ubuntu@tu-ec2-ip
```

#### Paso 2: Instalar Dependencias del Sistema

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git git-lfs
```

#### Paso 3: Clonar o Subir el Proyecto

**Opción A: Clonar desde Git**
```bash
cd /home/ubuntu
git clone <tu-repositorio-url> columna_vertebra_segmatacion
cd columna_vertebra_segmatacion
git lfs pull  # Descargar modelos grandes
```

**Opción B: Subir Archivos con SCP**
```bash
# Desde tu máquina local
scp -i tu-key.pem -r columna_vertebra_segmatacion ubuntu@tu-ec2-ip:/home/ubuntu/
```

#### Paso 4: Configurar la Aplicación

```bash
cd /home/ubuntu/columna_vertebra_segmatacion

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r segmentacion_app/requirements.txt
```

#### Paso 5: Configurar como Servicio systemd

**Crear archivo de servicio:**
```bash
sudo nano /etc/systemd/system/segmentacion.service
```

**Contenido del archivo:**
```ini
[Unit]
Description=Segmentacion Columna Vertebral API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/columna_vertebra_segmatacion
Environment="PATH=/home/ubuntu/columna_vertebra_segmatacion/venv/bin"
ExecStart=/home/ubuntu/columna_vertebra_segmatacion/venv/bin/uvicorn segmentacion_app.app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Activar el servicio:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable segmentacion
sudo systemctl start segmentacion
sudo systemctl status segmentacion
```

#### Paso 6: Configurar Security Group

1. Ve a la consola de AWS EC2
2. Selecciona tu instancia
3. Ve a "Security Groups"
4. Edita las reglas de entrada (Inbound rules)
5. Agrega una regla:
   - **Type:** Custom TCP
   - **Port:** 8000
   - **Source:** 0.0.0.0/0 (o tu IP específica)
   - **Description:** Segmentacion API

#### Paso 7: Verificar

```bash
# Ver logs
sudo journalctl -u segmentacion -f

# Probar localmente
curl http://localhost:8000/api/health

# Acceder desde Internet
# http://tu-ec2-ip-publica:8000
```

### Opción 2: Despliegue con Docker en EC2

#### Paso 1: Instalar Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
# Cerrar sesión y volver a conectar
```

#### Paso 2: Clonar/Subir Proyecto

```bash
cd /home/ubuntu
git clone <tu-repositorio-url> columna_vertebra_segmatacion
cd columna_vertebra_segmatacion
```

#### Paso 3: Construir y Ejecutar

```bash
docker build -t segmentacion-columna .
docker run -d \
  --name segmentacion \
  -p 8000:8000 \
  --restart unless-stopped \
  segmentacion-columna
```

#### Paso 4: Verificar

```bash
docker logs -f segmentacion
curl http://localhost:8000/api/health
```

### Comandos Útiles EC2

```bash
# Gestionar servicio systemd
sudo systemctl status segmentacion
sudo systemctl start segmentacion
sudo systemctl stop segmentacion
sudo systemctl restart segmentacion
sudo journalctl -u segmentacion -f

# Gestionar Docker
docker ps
docker logs -f segmentacion
docker restart segmentacion

# Ver uso de recursos
htop
df -h
free -h
```

## ☁️ Despliegue en Microsoft Azure

### Opción 1: Azure App Service

#### Paso 1: Preparar la Aplicación

```bash
# Crear archivo .deployment
echo [config] > .deployment
echo SCM_DO_BUILD_DURING_DEPLOYMENT=true >> .deployment

# Crear startup.sh
cat > startup.sh << 'EOF'
#!/bin/bash
cd /home/site/wwwroot
source venv/bin/activate
uvicorn segmentacion_app.app.main:app --host 0.0.0.0 --port 8000
EOF
chmod +x startup.sh
```

#### Paso 2: Desplegar con Azure CLI

```bash
# Instalar Azure CLI
# Windows: https://aka.ms/installazurecliwindows
# Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Crear grupo de recursos
az group create --name rg-segmentacion --location eastus

# Crear App Service Plan
az appservice plan create \
  --name plan-segmentacion \
  --resource-group rg-segmentacion \
  --sku B1 \
  --is-linux

# Crear Web App
az webapp create \
  --resource-group rg-segmentacion \
  --plan plan-segmentacion \
  --name segmentacion-columna \
  --runtime "PYTHON:3.10"

# Configurar startup
az webapp config set \
  --resource-group rg-segmentacion \
  --name segmentacion-columna \
  --startup-file "startup.sh"

# Desplegar código
az webapp deployment source config-zip \
  --resource-group rg-segmentacion \
  --name segmentacion-columna \
  --src segmentacion-columna.zip
```

### Opción 2: Azure Container Instances (ACI)

#### Paso 1: Construir y Subir Imagen a Azure Container Registry

```bash
# Crear Azure Container Registry
az acr create \
  --resource-group rg-segmentacion \
  --name acrsegmentacion \
  --sku Basic

# Login al ACR
az acr login --name acrsegmentacion

# Construir y subir imagen
az acr build \
  --registry acrsegmentacion \
  --image segmentacion-columna:latest \
  .
```

#### Paso 2: Crear Container Instance

```bash
az container create \
  --resource-group rg-segmentacion \
  --name segmentacion-columna \
  --image acrsegmentacion.azurecr.io/segmentacion-columna:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server acrsegmentacion.azurecr.io \
  --registry-username <acr-username> \
  --registry-password <acr-password> \
  --dns-name-label segmentacion-columna \
  --ports 8000
```

### Opción 3: Azure Virtual Machine

Similar a EC2, pero con Azure:

```bash
# Crear VM
az vm create \
  --resource-group rg-segmentacion \
  --name vm-segmentacion \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys

# Abrir puerto 8000
az vm open-port \
  --port 8000 \
  --resource-group rg-segmentacion \
  --name vm-segmentacion

# Conectar y seguir pasos de EC2
ssh azureuser@<vm-public-ip>
```

## 🌐 Despliegue en Google Cloud Platform (GCP)

### Opción 1: Google Cloud Run

#### Paso 1: Preparar Dockerfile

Asegúrate de que el Dockerfile esté optimizado para Cloud Run.

#### Paso 2: Construir y Subir Imagen

```bash
# Configurar proyecto
gcloud config set project tu-proyecto-id

# Habilitar APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Construir y subir
gcloud builds submit --tag gcr.io/tu-proyecto-id/segmentacion-columna

# Desplegar en Cloud Run
gcloud run deploy segmentacion-columna \
  --image gcr.io/tu-proyecto-id/segmentacion-columna \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Opción 2: Google Compute Engine (GCE)

Similar a EC2:

```bash
# Crear instancia
gcloud compute instances create segmentacion-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# Abrir puerto
gcloud compute firewall-rules create allow-segmentacion \
  --allow tcp:8000 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow segmentacion API"

# Conectar y seguir pasos de EC2
gcloud compute ssh segmentacion-vm --zone=us-central1-a
```

## 🔧 Configuración Avanzada

### Variables de Entorno

Crea un archivo `.env` (opcional):

```env
HOST=0.0.0.0
PORT=8000
DEBUG=False
MODEL_TYPE=deeplab_resnet50
DEVICE=cpu
```

### Configuración de Modelos

Los modelos se configuran en `segmentacion_app/app/config.py`:

```python
AVAILABLE_MODELS = {
    "deeplab_dense_decoder": {...},
    "unetplusplus_v2": {...},
    "deeplab_resnet50": {...}
}
```

### Tamaño de Imagen de Entrada

Modificar en `segmentacion_app/app/config.py`:

```python
INPUT_SIZE = (512, 512)  # Ajustar según necesidad
```

## 📡 API Endpoints

### `GET /`
Interfaz web principal para cargar y segmentar imágenes.

### `POST /api/segment`
Segmenta una imagen de radiografía.

**Parámetros (multipart/form-data):**
- `file`: Archivo de imagen (PNG, JPG, JPEG, DICOM)
- `model_type`: Tipo de modelo (opcional, default: "deeplab_resnet50")
  - Valores: `"deeplab_dense_decoder"`, `"unetplusplus_v2"`, `"deeplab_resnet50"`

**Respuesta:**
```json
{
  "success": true,
  "message": "Segmentación completada exitosamente",
  "model_used": "deeplab_resnet50",
  "original_image_url": "/static/original_xxx.png",
  "segmented_image_url": "/static/mask_xxx.png",
  "overlay_image_url": "/static/overlay_xxx.png",
  "classes_detected": ["F", "V", "T1"],
  "metrics": {
    "mean_iou": 0.1411,
    "mean_dice": 0.0911,
    "foreground_coverage": 9.98,
    "F_percentage": 90.02,
    "F_iou": 1.0000,
    "F_dice": 0.9423,
    "F_confidence": 0.9885,
    "V_percentage": 9.54,
    "V_iou": 0.2724,
    "V_dice": 0.1733,
    "V_confidence": 0.9522,
    "T1_percentage": 0.44,
    "T1_iou": 0.0099,
    "T1_dice": 0.0088,
    "T1_confidence": 0.7462
  }
}
```

### `GET /api/health`
Verifica el estado de la API y modelos.

**Parámetros de query (opcionales):**
- `model_type`: Tipo de modelo a verificar

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "classes": ["F", "V", "T1"],
  "model_type": "deeplab_resnet50"
}
```

## 📚 Documentación Adicional

Toda la documentación detallada está disponible en la carpeta [`docs/`](docs/):

- **[Guía de Despliegue EC2](docs/GUIA_DESPLIEGUE_EC2.md)** - Guía completa paso a paso para AWS EC2
- **[Guía de Instalación](docs/INSTALACION.md)** - Instrucciones detalladas de instalación
- **[Solución de Problemas](docs/SOLUCION_PROBLEMAS_UNET.md)** - Troubleshooting común
- **[Notas del Modelo](docs/NOTAS_MODELO.md)** - Notas técnicas sobre modelos

## 🐛 Solución de Problemas

> 💡 **Más ayuda:** Consulta [docs/SOLUCION_PROBLEMAS_UNET.md](docs/SOLUCION_PROBLEMAS_UNET.md) para problemas específicos con modelos U-Net++

### Error: "No module named 'torch._C'"

**Solución:**
```bash
# Desinstalar PyTorch
pip uninstall torch torchvision

# Reinstalar desde índice oficial
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Error: "Weights only load failed"

**Solución:** Ya está manejado en el código. Si persiste, verifica que el archivo del modelo esté completo (descarga con Git LFS).

### Error: "Ran out of input"

**Causa:** Archivo de modelo corrupto o incompleto.

**Solución:**
```bash
# Verificar tamaño del archivo
ls -lh models/*/*.pth

# Re-descargar con Git LFS
git lfs pull
```

### Error: "ModuleNotFoundError: No module named 'fastapi'"

**Solución:**
```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Reinstalar dependencias
pip install -r segmentacion_app/requirements.txt
```

### Puerto ya en uso

**Solución:**
```bash
# Cambiar puerto en config.py o usar variable de entorno
export PORT=8001
uvicorn segmentacion_app.app.main:app --port 8001
```

### Error de memoria en EC2

**Solución:**
- Usar instancia con más RAM (t3.large o superior)
- Reducir tamaño de entrada en `config.py`
- Usar modelo más pequeño

### Modelo no segmenta correctamente

**Verificar:**
1. Que las clases coincidan: `models/*/classes_*.json`
2. Que el modelo esté completamente descargado
3. Logs del servidor para errores específicos

## 📊 Modelos Disponibles

### DeepLabV3++ (Decoder Denso) - `deeplab_dense_decoder`
- **Arquitectura:** DeepLabV3+ con decoder denso tipo U-Net++
- **Características:** ASPP con atención, decoder de 4 capas, módulos de atención
- **Uso:** Balance entre precisión y complejidad

### U-Net++ v2 - `unetplusplus_v2`
- **Arquitectura:** U-Net++ optimizada
- **Características:** Skip connections densas, nested pathways
- **Uso:** Segmentación precisa con arquitectura U-Net

### DeepLabV3+ ResNet50 - `deeplab_resnet50`
- **Arquitectura:** DeepLabV3+ estándar de torchvision
- **Características:** Backbone ResNet50, ASPP estándar
- **Uso:** Modelo robusto y probado

## 🔄 Actualizar la Aplicación

### Local

```bash
git pull origin main
source venv/bin/activate  # Linux/Mac
pip install -r segmentacion_app/requirements.txt
```

### EC2 (systemd)

```bash
cd /home/ubuntu/columna_vertebra_segmatacion
git pull origin main
source venv/bin/activate
pip install -r segmentacion_app/requirements.txt
sudo systemctl restart segmentacion
```

### Docker

```bash
docker stop segmentacion
docker rm segmentacion
docker build -t segmentacion-columna .
docker run -d -p 8000:8000 --name segmentacion --restart unless-stopped segmentacion-columna
```

## 📝 Notas Importantes

- **Git LFS:** Los modelos grandes están en Git LFS. Asegúrate de tenerlo instalado y ejecutar `git lfs pull` después de clonar.
- **Modelos:** Los modelos se cargan bajo demanda. La primera carga puede tardar unos segundos.
- **Memoria:** Los modelos requieren ~2-4GB de RAM. Asegúrate de tener suficiente memoria disponible.
- **GPU:** Si tienes GPU NVIDIA, instala PyTorch con soporte CUDA para mejor rendimiento.

## 📄 Licencia

Este proyecto es para uso académico/investigación.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o soporte, abre un issue en el repositorio.
