# Schemas para segmentación
from pydantic import BaseModel
from typing import List, Optional


class SegmentationRequest(BaseModel):
    """Schema para la solicitud de segmentación"""
    pass  # Los archivos se envían como multipart/form-data


class SegmentationResponse(BaseModel):
    """Schema para la respuesta de segmentación"""
    success: bool
    message: str
    model_used: Optional[str] = None
    original_image_url: Optional[str] = None
    segmented_image_url: Optional[str] = None
    overlay_image_url: Optional[str] = None
    classes_detected: Optional[List[str]] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None

