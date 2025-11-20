# Modelo de segmentación DeepLabV3+
import os
import json
import zipfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

from ..config import (
    MODEL_ZIP_PATH,
    CLASSES_JSON_PATH,
    MODEL_EXTRACTED_DIR,
    INPUT_SIZE,
    NUM_CLASSES,
    AVAILABLE_MODELS
)


# ============================================================================
# MÓDULOS BASE DE LA ARQUITECTURA
# ============================================================================

class ConvBlock(nn.Module):
    """Bloque convolucional con BatchNorm y ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    """Arquitectura DeepLabV3+ personalizada"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.aspp = ASPP(512, 256)
        
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.out = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        aspp_out = self.aspp(enc4)
        low_level = self.decoder_conv1(enc1)
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        dec = torch.cat([aspp_up, low_level], dim=1)
        dec = self.decoder_conv2(dec)
        out = F.interpolate(dec, size=size, mode='bilinear', align_corners=True)
        
        return self.out(out)


class UNetPlusPlus(nn.Module):
    """Arquitectura U-Net++ personalizada"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv0_0 = ConvBlock(in_channels, 64)
        self.conv1_0 = ConvBlock(64, 128)
        self.conv2_0 = ConvBlock(128, 256)
        self.conv3_0 = ConvBlock(256, 512)
        self.conv4_0 = ConvBlock(512, 1024)
        
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv1_1 = ConvBlock(128 + 256, 128)
        self.conv2_1 = ConvBlock(256 + 512, 256)
        self.conv3_1 = ConvBlock(512 + 1024, 512)
        
        self.conv0_2 = ConvBlock(64 * 2 + 128, 64)
        self.conv1_2 = ConvBlock(128 * 2 + 256, 128)
        self.conv2_2 = ConvBlock(256 * 2 + 512, 256)
        
        self.conv0_3 = ConvBlock(64 * 3 + 128, 64)
        self.conv1_3 = ConvBlock(128 * 3 + 256, 128)
        
        self.conv0_4 = ConvBlock(64 * 4 + 128, 64)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        return self.out(x0_4)


class SegmentationModel:
    """Clase para cargar y usar modelos de segmentación"""
    
    def __init__(self, model_type: str = "deeplabv3plus"):
        """
        Inicializa el modelo de segmentación
        
        Args:
            model_type: Tipo de modelo a usar ('deeplabv3plus' o 'unetplusplus')
        """
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Modelo no disponible: {model_type}. Modelos disponibles: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_config = AVAILABLE_MODELS[model_type]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.model_loaded = False
        
    def _load_classes(self) -> list:
        """Carga las clases desde el archivo JSON del modelo"""
        try:
            classes_path = self.model_config["model_dir"] / self.model_config["classes_file"]
            if not classes_path.exists():
                # Fallback al archivo principal
                classes_path = CLASSES_JSON_PATH
            
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            return classes
        except Exception as e:
            print(f"Error cargando clases: {e}")
            return ["Background", "T1", "V"]
    
    def _find_model_file(self) -> Optional[Path]:
        """Busca el archivo del modelo"""
        model_dir = self.model_config["model_dir"]
        model_file = model_dir / self.model_config["model_file"]
        
        if model_file.exists():
            return model_file
        
        # Buscar cualquier archivo .pth en el directorio
        for file_path in model_dir.glob("*.pth"):
            if file_path.is_file():
                return file_path
        
        return None
    
    def load_model(self):
        """Carga el modelo de segmentación"""
        if self.model_loaded:
            return
        
        try:
            model_path = self._find_model_file()
            
            if model_path is None:
                model_dir = self.model_config["model_dir"]
                raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_dir}")
            
            print(f"Cargando modelo {self.model_config['name']} desde {model_path}...")
            
            # Intentar cargar como modelo PyTorch
            try:
                # PyTorch 2.6+ cambió el default de weights_only a True por seguridad
                # Como estos son nuestros modelos propios y confiables, usamos weights_only=False
                # También agregamos safe globals para numpy arrays que pueden estar en el checkpoint
                try:
                    # Intentar agregar safe globals para numpy (si está disponible en la versión)
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                except (AttributeError, ImportError):
                    # Si no está disponible, continuamos sin ello
                    pass
                
                # Cargar checkpoint con weights_only=False (confiamos en nuestros modelos)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Intentar diferentes estructuras de checkpoint
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        # Modelo completo guardado
                        self.model = checkpoint['model']
                        if hasattr(self.model, 'eval'):
                            self.model.eval()
                    elif 'model_state_dict' in checkpoint:
                        # Solo state_dict - necesitamos reconstruir la arquitectura
                        state_dict = checkpoint['model_state_dict']
                        
                        # Determinar qué arquitectura usar
                        first_key = list(state_dict.keys())[0] if state_dict else ""
                        architecture_name = self.model_config["architecture"]
                        
                        # Detectar arquitectura por las keys del state_dict
                        is_deeplab = 'enc1' in first_key and 'aspp' in first_key and 'decoder_conv' in first_key
                        is_unetpp = 'conv0_0' in first_key or 'conv0_1' in first_key
                        
                        print(f"Reconstruyendo arquitectura {architecture_name}...")
                        
                        if architecture_name == "DeepLabV3Plus" or is_deeplab:
                            try:
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3+: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        elif architecture_name == "UNetPlusPlus" or is_unetpp:
                            try:
                                self.model = UNetPlusPlus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(
                                    f"Error cargando modelo UNet++: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        else:
                            # Intentar con arquitecturas estándar de torchvision
                            print("Reconstruyendo arquitectura DeepLabV3+ estándar...")
                            # Intentar con ResNet50 primero (más común)
                            try:
                                from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
                                self.model = deeplabv3_resnet50(
                                    num_classes=NUM_CLASSES,
                                    pretrained_backbone=False
                                )
                                self.model.load_state_dict(state_dict, strict=False)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ ResNet50 cargado exitosamente")
                            except Exception as e1:
                                print(f"Error con ResNet50: {e1}")
                                print("Intentando con ResNet101...")
                                try:
                                    self.model = deeplabv3_resnet101(
                                        num_classes=NUM_CLASSES,
                                        pretrained_backbone=False
                                    )
                                    self.model.load_state_dict(state_dict, strict=False)
                                    self.model = self.model.to(self.device)
                                    self.model.eval()
                                    print("Modelo DeepLabV3+ ResNet101 cargado exitosamente")
                                except Exception as e2:
                                    raise RuntimeError(
                                        f"No se pudo cargar el modelo con ResNet50 ni ResNet101.\n"
                                        f"El modelo parece usar una arquitectura personalizada.\n"
                                        f"Error ResNet50: {str(e1)[:200]}\n"
                                        f"Error ResNet101: {str(e2)[:200]}"
                                    )
                    elif 'state_dict' in checkpoint:
                        # Formato alternativo con 'state_dict'
                        state_dict = checkpoint['state_dict']
                        first_key = list(state_dict.keys())[0] if state_dict else ""
                        is_custom_arch = 'enc1' in first_key or 'aspp' in first_key or 'decoder_conv' in first_key
                        
                        if is_custom_arch:
                            print("Reconstruyendo arquitectura DeepLabV3+ personalizada (formato state_dict)...")
                            try:
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ personalizado cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(f"Error cargando modelo DeepLabV3+ personalizado: {e}")
                        else:
                            print("Reconstruyendo arquitectura DeepLabV3+ estándar (formato state_dict)...")
                            try:
                                from torchvision.models.segmentation import deeplabv3_resnet50
                                self.model = deeplabv3_resnet50(
                                    num_classes=NUM_CLASSES,
                                    pretrained_backbone=False
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ ResNet50 cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(f"Error cargando modelo: {e}")
                    else:
                        # Si es un dict pero no tiene las keys esperadas, asumir que es el modelo completo
                        # Esto es poco probable pero lo manejamos
                        raise ValueError(f"Formato de checkpoint no reconocido. Keys encontradas: {list(checkpoint.keys())}")
                else:
                    # Si no es dict, asumir que es el modelo completo
                    self.model = checkpoint
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                
                # Verificar que el modelo se cargó correctamente
                if not hasattr(self.model, 'forward'):
                    raise RuntimeError("El modelo cargado no tiene método 'forward'")
                
                self.model_loaded = True
                print(f"Modelo cargado exitosamente en {self.device}")
                
            except Exception as e:
                print(f"Error cargando modelo PyTorch: {e}")
                print("Intentando otros formatos...")
                # Aquí podrías agregar lógica para otros formatos (TensorFlow, ONNX, etc.)
                raise
        
        except Exception as e:
            print(f"Error en load_model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa la imagen para el modelo"""
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar
        image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        
        # Convertir a numpy array y normalizar
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convertir de HWC a CHW y agregar batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def postprocess_prediction(self, prediction: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocesa la predicción del modelo"""
        # Obtener la clase predicha para cada pixel
        if prediction.dim() > 3:
            prediction = prediction.squeeze(0)
        
        if prediction.dim() == 3:
            # Si tiene dimensiones [C, H, W], tomar argmax
            pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()
        else:
            # Si ya es [H, W]
            pred_mask = prediction.cpu().numpy()
        
        # Redimensionar a tamaño original
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_mask
    
    def predict(self, image: Image.Image, return_probs: bool = False):
        """
        Realiza la predicción de segmentación
        
        Args:
            image: Imagen a segmentar
            return_probs: Si True, retorna también las probabilidades
            
        Returns:
            mask: Máscara de segmentación (H, W) o tupla (mask, probs) si return_probs=True
        """
        if not self.model_loaded:
            self.load_model()
        
        original_size = image.size
        
        # Preprocesar
        input_tensor = self.preprocess_image(image)
        
        # Predecir
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Si el output es un dict (formato estándar de torchvision segmentation models)
            if isinstance(output, dict):
                output = output['out']
        
        # Obtener probabilidades antes del argmax
        probs = torch.softmax(output, dim=1) if output.dim() > 2 else None
        
        # Postprocesar
        mask = self.postprocess_prediction(output, original_size)
        
        if return_probs and probs is not None:
            # Redimensionar probabilidades al tamaño original
            probs_np = probs.squeeze(0).cpu().numpy()  # [C, H, W]
            probs_resized = np.zeros((probs_np.shape[0], original_size[1], original_size[0]))
            for c in range(probs_np.shape[0]):
                probs_resized[c] = cv2.resize(
                    probs_np[c],
                    original_size,
                    interpolation=cv2.INTER_LINEAR
                )
            return mask, probs_resized
        
        return mask
    
    def create_visualization(self, original_image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Crea una visualización de la segmentación superpuesta sobre la imagen original"""
        # Convertir imagen original a numpy y asegurar tipo uint8
        img_array = np.array(original_image.convert('RGB')).astype(np.uint8)
        
        # Asegurar que mask sea int y tenga valores válidos
        mask = mask.astype(np.int32)
        mask = np.clip(mask, 0, len(self.classes) - 1)
        
        # Redimensionar mask si es necesario para que coincida con la imagen
        if mask.shape[:2] != img_array.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img_array.shape[1], img_array.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # Crear colores más distintivos para cada clase (uint8)
        # T1 en azul brillante, V (columna) en verde/amarillo para mejor diferenciación
        colors = np.array([
            [0, 0, 0],           # Background - negro
            [0, 150, 255],       # T1 - azul brillante (más visible)
            [255, 200, 0],       # V - amarillo/naranja (diferente a T1)
        ], dtype=np.uint8)
        
        # Crear imagen de segmentación coloreada
        # Asegurar que mask tenga la forma correcta para indexar
        if mask.ndim == 2:
            colored_mask = colors[mask]
        else:
            # Si mask tiene más dimensiones, tomar solo la primera
            colored_mask = colors[mask.reshape(-1)].reshape(*mask.shape, 3)
        
        # Asegurar que colored_mask sea uint8
        colored_mask = colored_mask.astype(np.uint8)
        
        # Verificar que las dimensiones coincidan
        if colored_mask.shape[:2] != img_array.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (img_array.shape[1], img_array.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Superponer con transparencia (ambos deben ser uint8 del mismo tamaño)
        # Asegurar que ambos arrays sean uint8 y tengan las mismas dimensiones
        img_uint8 = img_array.astype(np.uint8)
        mask_uint8 = colored_mask.astype(np.uint8)
        
        # Verificar dimensiones una última vez
        if img_uint8.shape != mask_uint8.shape:
            mask_uint8 = cv2.resize(mask_uint8, (img_uint8.shape[1], img_uint8.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Superponer con transparencia
        overlay = cv2.addWeighted(img_uint8, 0.6, mask_uint8, 0.4, 0)
        
        return Image.fromarray(overlay)
    
    def calculate_metrics(self, mask: np.ndarray, probs: Optional[np.ndarray] = None) -> dict:
        """
        Calcula métricas de la segmentación incluyendo IoU y Dice estimados basados en confianza
        
        Args:
            mask: Máscara de segmentación (H, W) con valores de clase
            probs: Probabilidades del modelo [C, H, W] (opcional)
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {}
        
        # Calcular distribución de clases
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Porcentaje de cada clase
        class_percentages = {}
        for cls, count in zip(unique_classes, counts):
            if cls < len(self.classes):
                class_name = self.classes[int(cls)]
                percentage = float(count / total_pixels * 100)
                class_percentages[class_name] = percentage
                metrics[f"{class_name}_percentage"] = percentage
                metrics[f"{class_name}_pixels"] = int(count)
        
        # Calcular cobertura total (todo excepto background)
        background_pixels = counts[unique_classes == 0].sum() if 0 in unique_classes else 0
        foreground_pixels = total_pixels - background_pixels
        metrics["foreground_coverage"] = float(foreground_pixels / total_pixels * 100)
        
        # Calcular métricas básicas
        metrics["total_classes_detected"] = len(unique_classes)
        metrics["total_pixels"] = int(total_pixels)
        
        # Calcular IoU y Dice estimados basados en confianza del modelo
        if probs is not None and probs.shape[0] == len(self.classes):
            # probs es [C, H, W]
            # Calcular confianza promedio por clase
            for c in range(len(self.classes)):
                if c < probs.shape[0]:
                    class_name = self.classes[c]
                    # Máscara binaria para esta clase
                    pred_mask = (mask == c)
                    
                    # Probabilidad promedio en los píxeles predichos como esta clase
                    if pred_mask.sum() > 0:
                        avg_confidence = float(probs[c][pred_mask].mean())
                        metrics[f"{class_name}_confidence"] = avg_confidence
                        
                        # IoU estimado basado en confianza
                        # Si la confianza es alta, el IoU estimado es alto
                        # Fórmula: IoU_estimado = confianza * (píxeles_predichos / total_píxeles)
                        predicted_pixels = pred_mask.sum()
                        iou_estimated = avg_confidence * (predicted_pixels / total_pixels) * len(self.classes)
                        metrics[f"{class_name}_iou_estimated"] = float(min(iou_estimated, 1.0))
                        
                        # Dice estimado: 2 * confianza * recall_estimado / (confianza + recall_estimado)
                        recall_estimated = predicted_pixels / total_pixels if total_pixels > 0 else 0
                        dice_estimated = (2 * avg_confidence * recall_estimated) / (avg_confidence + recall_estimated + 1e-10)
                        metrics[f"{class_name}_dice_estimated"] = float(min(dice_estimated, 1.0))
            
            # Calcular IoU y Dice promedio (sin background)
            ious_estimated = [metrics.get(f"{self.classes[c]}_iou_estimated", 0.0) 
                            for c in range(1, len(self.classes))]
            dices_estimated = [metrics.get(f"{self.classes[c]}_dice_estimated", 0.0) 
                             for c in range(1, len(self.classes))]
            
            if ious_estimated:
                metrics["mean_iou_estimated"] = float(np.mean(ious_estimated))
            if dices_estimated:
                metrics["mean_dice_estimated"] = float(np.mean(dices_estimated))
        
        # Calcular entropía de la distribución
        if len(unique_classes) > 1:
            non_bg_classes = [c for c in unique_classes if c != 0]
            if len(non_bg_classes) > 0:
                class_probs = [counts[unique_classes == c][0] / total_pixels for c in non_bg_classes]
                entropy = -sum(p * np.log2(p + 1e-10) for p in class_probs)
                metrics["prediction_entropy"] = float(entropy)
        
        return metrics
    
    def improve_t1_segmentation(self, mask: np.ndarray, probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mejora la segmentación de T1 usando post-procesamiento
        
        Args:
            mask: Máscara de segmentación original
            probs: Probabilidades del modelo [C, H, W] (opcional)
            
        Returns:
            Máscara mejorada
        """
        improved_mask = mask.copy()
        
        # T1 es clase 1 (índice en la lista de clases)
        t1_class = 1
        
        # Verificar que T1 existe en las clases
        if t1_class >= len(self.classes) or self.classes[t1_class] != "T1":
            return improved_mask
        
        # Crear máscara binaria de T1
        t1_mask = (improved_mask == t1_class).astype(np.uint8)
        
        # Operaciones morfológicas para mejorar T1
        # 1. Eliminar pequeños ruidos (opening)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        t1_cleaned = cv2.morphologyEx(t1_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. Rellenar huecos pequeños (closing)
        t1_filled = cv2.morphologyEx(t1_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # 3. Si tenemos probabilidades, usar umbral de confianza
        if probs is not None and t1_class < probs.shape[0]:
            t1_probs = probs[t1_class]
            # Aplicar umbral de confianza (solo mantener píxeles con alta confianza)
            confidence_threshold = 0.5
            high_confidence = (t1_probs > confidence_threshold).astype(np.uint8)
            # Combinar con máscara morfológica
            t1_filled = np.logical_and(t1_filled, high_confidence).astype(np.uint8)
        
        # Actualizar máscara mejorada
        improved_mask[t1_filled == 1] = t1_class
        # Eliminar T1 de baja confianza (donde había T1 pero ahora no)
        improved_mask[(t1_mask == 1) & (t1_filled == 0)] = 0
        
        return improved_mask

