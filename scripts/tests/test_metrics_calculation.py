"""Script para probar el cálculo de métricas"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.model.segmentation_model import SegmentationModel
from PIL import Image
import numpy as np

print("Probando cálculo de métricas con deeplab_resnet50...")
model = SegmentationModel('deeplab_resnet50')
model.load_model()

# Crear imagen de prueba
test_img = Image.new('RGB', (256, 512), color='white')

# Predecir
mask, probs = model.predict(test_img, return_probs=True)
print(f"Shape de mask: {mask.shape}")
print(f"Shape de probs: {probs.shape}")
print(f"Valores únicos en mask: {np.unique(mask)}")

# Calcular métricas
metrics = model.calculate_metrics(mask, probs)

print("\nMétricas calculadas:")
print(f"  IoU Promedio: {metrics.get('mean_iou', 0):.4f}")
print(f"  Dice Promedio: {metrics.get('mean_dice', 0):.4f}")
print(f"  Cobertura Foreground: {metrics.get('foreground_coverage', 0):.2f}%")

print("\nMétricas por clase:")
for cls in ['F', 'V', 'T1']:
    if f'{cls}_iou' in metrics:
        print(f"  {cls}:")
        print(f"    Porcentaje: {metrics.get(f'{cls}_percentage', 0):.2f}%")
        print(f"    IoU: {metrics.get(f'{cls}_iou', 0):.4f}")
        print(f"    Dice: {metrics.get(f'{cls}_dice', 0):.4f}")
        print(f"    Confianza: {metrics.get(f'{cls}_confidence', 0):.4f}")

