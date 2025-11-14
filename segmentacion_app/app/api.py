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

from .config import STATIC_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from .model.segmentation_model import SegmentationModel
from .schemas.segmentation import SegmentationResponse

router = APIRouter()

# Instancia global del modelo
segmentation_model = None


def get_model():
    """Obtiene o inicializa el modelo de segmentación"""
    global segmentation_model
    if segmentation_model is None:
        segmentation_model = SegmentationModel()
        segmentation_model.load_model()
    return segmentation_model


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
async def segment_image(file: UploadFile = File(...)):
    """
    Endpoint para segmentar una imagen de radiografía
    
    Args:
        file: Archivo de imagen a segmentar
        
    Returns:
        SegmentationResponse con las URLs de las imágenes procesadas
    """
    try:
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
        model = get_model()
        mask = model.predict(image)
        
        # Crear visualizaciones
        segmented_image = model.create_visualization(image, mask)
        
        # Guardar imágenes en static
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())
        
        # Guardar imagen original
        original_path = STATIC_DIR / f"original_{unique_id}.png"
        image.save(original_path)
        original_url = f"/static/original_{unique_id}.png"
        
        # Guardar imagen segmentada (solo máscara)
        mask_image = Image.fromarray((mask * 85).astype(np.uint8))  # Escalar para visualización
        mask_path = STATIC_DIR / f"mask_{unique_id}.png"
        mask_image.save(mask_path)
        mask_url = f"/static/mask_{unique_id}.png"
        
        # Guardar imagen superpuesta
        overlay_path = STATIC_DIR / f"overlay_{unique_id}.png"
        segmented_image.save(overlay_path)
        overlay_url = f"/static/overlay_{unique_id}.png"
        
        # Detectar clases presentes
        unique_classes = np.unique(mask)
        classes_detected = [model.classes[int(cls)] for cls in unique_classes if int(cls) < len(model.classes)]
        
        return SegmentationResponse(
            success=True,
            message="Segmentación completada exitosamente",
            original_image_url=original_url,
            segmented_image_url=mask_url,
            overlay_image_url=overlay_url,
            classes_detected=classes_detected
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
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model.model_loaded,
            "device": str(model.device),
            "classes": model.classes
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

