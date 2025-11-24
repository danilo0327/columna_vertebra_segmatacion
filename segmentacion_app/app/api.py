# Endpoints de la API
import os
import uuid
import numpy as np
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io

from .config import STATIC_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE, AVAILABLE_MODELS
from .model.segmentation_model import SegmentationModel
from .schemas.segmentation import SegmentationResponse

router = APIRouter()

# Cache de modelos cargados
models_cache = {}


def get_model(model_type: str = "deeplabv3plus"):
    """
    Obtiene o inicializa el modelo de segmentación
    
    Args:
        model_type: Tipo de modelo ('deeplabv3plus' o 'unetplusplus')
    
    Returns:
        Instancia del modelo cargado
    """
    if model_type not in models_cache:
        models_cache[model_type] = SegmentationModel(model_type=model_type)
        models_cache[model_type].load_model()
    return models_cache[model_type]


@router.get("/", response_class=HTMLResponse)
async def home():
    """Endpoint principal que muestra la interfaz web"""
    from .main import get_app
    app = get_app()
    template_path = Path(__file__).parent / "templates" / "index.html"
    
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>Segmentación de Columna Vertebral</title></head>
            <body>
                <h1>Segmentación de Columna Vertebral</h1>
                <p>La interfaz web está en desarrollo. Use el endpoint /api/segment para segmentar imágenes.</p>
            </body>
        </html>
        """


@router.post("/api/segment", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(...),
    model_type: str = Form("deeplabv3plus")
):
    """
    Endpoint para segmentar una imagen de radiografía
    
    Args:
        file: Archivo de imagen a segmentar
        model_type: Tipo de modelo a usar ('deeplabv3plus' o 'unetplusplus')
        
    Returns:
        SegmentationResponse con las URLs de las imágenes procesadas y métricas
    """
    try:
        # Validar modelo
        if model_type not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Modelo no disponible: {model_type}. Modelos disponibles: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Validar extensión del archivo
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato de archivo no permitido. Formatos permitidos: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Leer el archivo
        contents = await file.read()
        
        # Validar tamaño
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo demasiado grande. Tamaño máximo: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Cargar imagen
        try:
            image = Image.open(io.BytesIO(contents))
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al cargar la imagen: {str(e)}")
        
        # Obtener modelo y realizar segmentación
        try:
            model = get_model(model_type)
        except Exception as e:
            error_msg = f"Error al cargar el modelo {model_type}: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        try:
            mask, probs = model.predict(image, return_probs=True)
        except Exception as e:
            error_msg = f"Error al procesar la imagen con el modelo {model_type}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Mejorar segmentación de T1
        mask_improved = model.improve_t1_segmentation(mask, probs)
        
        # Calcular métricas (con probabilidades para IoU/Dice estimados)
        metrics = model.calculate_metrics(mask_improved, probs)
        
        # Crear visualizaciones con máscara mejorada
        segmented_image = model.create_visualization(image, mask_improved)
        
        # Guardar imágenes en static
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())
        
        # Guardar imagen original
        original_path = STATIC_DIR / f"original_{unique_id}.png"
        image.save(original_path)
        original_url = f"/static/original_{unique_id}.png"
        
        # Guardar imagen segmentada (solo máscara) - usar máscara mejorada
        mask_image = Image.fromarray((mask_improved * 85).astype(np.uint8))  # Escalar para visualización
        mask_path = STATIC_DIR / f"mask_{unique_id}.png"
        mask_image.save(mask_path)
        mask_url = f"/static/mask_{unique_id}.png"
        
        # Detectar clases presentes (usar máscara mejorada)
        unique_classes = np.unique(mask_improved)
        
        # Guardar imagen superpuesta
        overlay_path = STATIC_DIR / f"overlay_{unique_id}.png"
        segmented_image.save(overlay_path)
        overlay_url = f"/static/overlay_{unique_id}.png"
        
        classes_detected = [model.classes[int(cls)] for cls in unique_classes if int(cls) < len(model.classes)]
        
        return SegmentationResponse(
            success=True,
            message="Segmentación completada exitosamente",
            model_used=AVAILABLE_MODELS[model_type]["name"],
            original_image_url=original_url,
            segmented_image_url=mask_url,
            overlay_image_url=overlay_url,
            classes_detected=classes_detected,
            metrics=metrics
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return SegmentationResponse(
            success=False,
            message="Error al procesar la imagen",
            error=str(e)
        )


@router.get("/api/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    try:
        # Verificar todos los modelos disponibles
        models_status = {}
        for model_type in AVAILABLE_MODELS.keys():
            try:
                model = get_model(model_type)
                models_status[model_type] = {
                    "loaded": model.model_loaded,
                    "device": str(model.device),
                    "classes": model.classes
                }
            except Exception as e:
                models_status[model_type] = {
                    "loaded": False,
                    "error": str(e)
                }
        
        return {
            "status": "healthy",
            "available_models": list(AVAILABLE_MODELS.keys()),
            "models_status": models_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/api/models")
async def list_models():
    """Endpoint para listar modelos disponibles"""
    return {
        "models": {
            key: {
                "name": config["name"],
                "architecture": config["architecture"]
            }
            for key, config in AVAILABLE_MODELS.items()
        }
    }

