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
    NUM_CLASSES
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


class SegmentationModel:
    """Clase para cargar y usar el modelo de segmentación DeepLabV3+"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.model_loaded = False
        
    def _load_classes(self) -> list:
        """Carga las clases desde el archivo JSON"""
        try:
            with open(CLASSES_JSON_PATH, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            return classes
        except Exception as e:
            print(f"Error cargando clases: {e}")
            return ["Background", "T1", "V"]
    
    def _extract_model_if_needed(self):
        """Extrae el modelo del ZIP si no está extraído"""
        if MODEL_EXTRACTED_DIR.exists() and any(MODEL_EXTRACTED_DIR.iterdir()):
            return
        
        print(f"Extrayendo modelo desde {MODEL_ZIP_PATH}...")
        MODEL_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(MODEL_EXTRACTED_DIR)
            print(f"Modelo extraído en {MODEL_EXTRACTED_DIR}")
        except Exception as e:
            print(f"Error extrayendo modelo: {e}")
            raise
    
    def _find_model_file(self) -> Optional[Path]:
        """Busca el archivo del modelo (.pth, .pt, .h5, etc.)"""
        model_extensions = ['.pth', '.pt', '.pkl', '.h5', '.hdf5', '.onnx']
        
        for ext in model_extensions:
            for file_path in MODEL_EXTRACTED_DIR.rglob(f'*{ext}'):
                if file_path.is_file():
                    return file_path
        
        # Si no encuentra, busca cualquier archivo que pueda ser el modelo
        for file_path in MODEL_EXTRACTED_DIR.rglob('*'):
            if file_path.is_file() and file_path.suffix:
                return file_path
        
        return None
    
    def load_model(self):
        """Carga el modelo de segmentación"""
        if self.model_loaded:
            return
        
        try:
            self._extract_model_if_needed()
            model_path = self._find_model_file()
            
            if model_path is None:
                raise FileNotFoundError(f"No se encontró el archivo del modelo en {MODEL_EXTRACTED_DIR}")
            
            print(f"Cargando modelo desde {model_path}...")
            
            # Intentar cargar como modelo PyTorch
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
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
                        
                        # Verificar si es una arquitectura personalizada (tiene enc1, aspp, etc.)
                        first_key = list(state_dict.keys())[0] if state_dict else ""
                        is_custom_arch = 'enc1' in first_key or 'aspp' in first_key or 'decoder_conv' in first_key
                        
                        if is_custom_arch:
                            # Arquitectura personalizada DeepLabV3+
                            print("Reconstruyendo arquitectura DeepLabV3+ personalizada...")
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
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3+ personalizado: {str(e)}\n"
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
    
    def predict(self, image: Image.Image) -> np.ndarray:
        """Realiza la predicción de segmentación"""
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
        
        # Postprocesar
        mask = self.postprocess_prediction(output, original_size)
        
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
        
        # Crear colores para cada clase (uint8)
        colors = np.array([
            [0, 0, 0],        # Background - negro
            [255, 0, 0],      # T1 - rojo
            [0, 255, 0],      # V - verde
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

