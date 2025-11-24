# Configuración de la aplicación
import os
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).parent.parent.parent
APP_DIR = Path(__file__).parent

# Configuración de rutas
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

# Configuración del modelo
MODEL_ZIP_PATH = BASE_DIR / "deeplabv3plus_20251114_040131.zip"
CLASSES_JSON_PATH = BASE_DIR / "classes_deeplabv3plus.json"
MODEL_EXTRACTED_DIR = MODEL_DIR / "deeplabv3plus"

# Configuración de modelos disponibles
AVAILABLE_MODELS = {
    "deeplab_hybrid": {
        "name": "DeepLabV3++ (Decoder Denso)",
        "model_dir": MODEL_DIR / "deeplab_hybrid",
        "model_file": "DeepLabV3pp_best.pth",
        "classes_file": "classes_deeplab_hybrid.json",
        "architecture": "DeepLabV3Plus"
    },
    "unetplusplus_v2": {
        "name": "U-Net++ v2",
        "model_dir": MODEL_DIR / "unetplusplus_v2",
        "model_file": "u_netplusplus_best.pth",
        "classes_file": "classes_unetplusplus_v2.json",
        "architecture": "UNetPlusPlus"
    },
    "deeplab_resnet50": {
        "name": "DeepLabV3+ ResNet50",
        "model_dir": MODEL_DIR / "deeplab_resnet50",
        "model_file": "model_spine_t1_deeplabv3.pth",
        "classes_file": "spine_t1_classes.json",
        "architecture": "DeepLabV3Plus"
    }
}

# Configuración de la aplicación
APP_NAME = "Segmentación de Columna Vertebral"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Configuración del servidor
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Configuración de archivos
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Configuración de segmentación
INPUT_SIZE = (512, 512)  # Tamaño de entrada del modelo
NUM_CLASSES = 3  # Background, T1, V

