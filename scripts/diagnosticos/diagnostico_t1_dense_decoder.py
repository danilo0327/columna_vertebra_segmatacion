"""Script para diagnosticar por qué T1 no se segmenta con deeplab_dense_decoder"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.model.segmentation_model import SegmentationModel
from PIL import Image
import numpy as np
import torch
import cv2

print("=" * 70)
print("Diagnóstico de segmentación T1 - DeepLabV3PlusDenseDecoder")
print("=" * 70)

# Cargar modelo
print("\n1. Cargando modelo...")
model = SegmentationModel('deeplab_dense_decoder')
model.load_model()
print(f"   ✓ Modelo cargado: {model.model_config['name']}")
print(f"   ✓ Clases: {model.classes}")
print(f"   ✓ Índice T1: {model._get_class_index('T1')}")
print(f"   ✓ Índice V: {model._get_class_index('V')}")
print(f"   ✓ Índice F: {model._get_class_index('F')}")

# Crear imagen de prueba (blanco)
print("\n2. Creando imagen de prueba (256x256, blanco)...")
test_img = Image.new('RGB', (256, 256), color='white')

# Preprocesar
print("\n3. Preprocesando imagen...")
input_tensor = model.preprocess_image(test_img)
print(f"   ✓ Shape del tensor: {input_tensor.shape}")

# Predecir
print("\n4. Realizando predicción...")
with torch.no_grad():
    output = model.model(input_tensor)
    if isinstance(output, dict):
        output = output['out']
    print(f"   ✓ Shape del output: {output.shape}")
    
    # Obtener probabilidades
    probs = torch.softmax(output, dim=1)
    probs_np = probs.squeeze(0).cpu().numpy()  # [C, H, W]
    print(f"   ✓ Shape de probabilidades: {probs_np.shape}")
    print(f"   ✓ Número de clases: {probs_np.shape[0]}")

# Analizar probabilidades por clase
print("\n5. Análisis de probabilidades:")
t1_idx = model._get_class_index("T1")
v_idx = model._get_class_index("V")
f_idx = model._get_class_index("F")

for c in range(probs_np.shape[0]):
    class_name = model.classes[c] if c < len(model.classes) else f"Class_{c}"
    class_probs = probs_np[c]
    print(f"\n   Clase {c} ({class_name}):")
    print(f"     - Min: {class_probs.min():.6f}")
    print(f"     - Max: {class_probs.max():.6f}")
    print(f"     - Mean: {class_probs.mean():.6f}")
    print(f"     - Median: {np.median(class_probs):.6f}")
    print(f"     - Píxeles > 0.1: {(class_probs > 0.1).sum()}")
    print(f"     - Píxeles > 0.05: {(class_probs > 0.05).sum()}")
    print(f"     - Píxeles > 0.01: {(class_probs > 0.01).sum()}")

# Verificar post-procesamiento
print("\n6. Verificando post-procesamiento...")
mask = model.postprocess_prediction(output, (256, 256))
unique_classes, counts = np.unique(mask, return_counts=True)
print(f"   ✓ Clases detectadas en máscara: {unique_classes}")
print(f"   ✓ Conteos: {dict(zip(unique_classes, counts))}")

# Verificar improve_t1_segmentation
print("\n7. Verificando improve_t1_segmentation...")
# Redimensionar probabilidades al tamaño original
probs_resized = np.zeros((probs_np.shape[0], 256, 256))
for c in range(probs_np.shape[0]):
    probs_resized[c] = cv2.resize(
        probs_np[c],
        (256, 256),
        interpolation=cv2.INTER_LINEAR
    )

mask_improved = model.improve_t1_segmentation(mask, probs_resized)
unique_classes_improved, counts_improved = np.unique(mask_improved, return_counts=True)
print(f"   ✓ Clases después de improve_t1: {unique_classes_improved}")
print(f"   ✓ Conteos: {dict(zip(unique_classes_improved, counts_improved))}")

# Verificar umbrales
if t1_idx is not None and t1_idx < probs_resized.shape[0]:
    t1_probs = probs_resized[t1_idx]
    print(f"\n8. Análisis específico de T1:")
    print(f"   - Probabilidad máxima de T1: {t1_probs.max():.6f}")
    print(f"   - Probabilidad promedio de T1: {t1_probs.mean():.6f}")
    print(f"   - Píxeles con T1 > 0.05: {(t1_probs > 0.05).sum()}")
    print(f"   - Píxeles con T1 > 0.04: {(t1_probs > 0.04).sum()}")
    print(f"   - Píxeles con T1 > 0.01: {(t1_probs > 0.01).sum()}")
    
    # Verificar condiciones del post-procesamiento
    max_probs = np.max(probs_resized, axis=0)
    t1_relative = t1_probs / (max_probs + 1e-8)
    print(f"   - Probabilidad relativa máxima: {t1_relative.max():.6f}")
    print(f"   - Píxeles con T1 relativa > 0.15: {(t1_relative > 0.15).sum()}")

print("\n" + "=" * 70)

