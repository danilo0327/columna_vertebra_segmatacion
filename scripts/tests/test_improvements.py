"""Script para probar las mejoras de segmentación"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.model.segmentation_model import SegmentationModel
from PIL import Image
import numpy as np

# Cargar modelo
print("Cargando modelo...")
model = SegmentationModel('deeplab_hybrid')
model.load_model()

# Crear imagen de prueba
test_img = Image.new('RGB', (512, 512), color='white')

print("\nProbando predicción con mejoras...")
mask, probs = model.predict(test_img, return_probs=True)
print(f"Valores únicos antes de mejoras: {np.unique(mask)}")

# Aplicar mejoras
mask_v_improved = model.improve_v_segmentation(mask, probs)
mask_final = model.improve_t1_segmentation(mask_v_improved, probs)

print(f"Valores únicos después de mejoras: {np.unique(mask_final)}")

unique, counts = np.unique(mask_final, return_counts=True)
print("\nDistribución de clases en máscara mejorada:")
for u, c in zip(unique, counts):
    class_name = model.classes[u] if u < len(model.classes) else "unknown"
    percentage = c / mask_final.size * 100
    print(f"  Clase {u} ({class_name}): {c} píxeles ({percentage:.2f}%)")

# Calcular métricas
metrics = model.calculate_metrics(mask_final, probs)
print("\nMétricas mejoradas:")
print(f"  IoU Promedio (Estimado): {metrics.get('mean_iou_estimated', 0):.4f}")
print(f"  Dice Promedio (Estimado): {metrics.get('mean_dice_estimated', 0):.4f}")
print(f"  Cobertura Foreground: {metrics.get('foreground_coverage', 0):.2f}%")

# Métricas por clase
for class_name in ['T1', 'V']:
    if f"{class_name}_percentage" in metrics:
        print(f"\n{class_name}:")
        print(f"  Porcentaje: {metrics[f'{class_name}_percentage']:.2f}%")
        print(f"  IoU (Estimado): {metrics.get(f'{class_name}_iou_estimated', 0):.4f}")
        print(f"  Dice (Estimado): {metrics.get(f'{class_name}_dice_estimated', 0):.4f}")
        print(f"  Confianza: {metrics.get(f'{class_name}_confidence', 0):.4f}")

