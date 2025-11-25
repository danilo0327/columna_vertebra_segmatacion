"""Script para probar las mejoras de segmentación de T1"""
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

print("\nProbando predicción mejorada...")
mask, probs = model.predict(test_img, return_probs=True)
print(f"Valores únicos antes de improve_t1: {np.unique(mask)}")

mask_improved = model.improve_t1_segmentation(mask, probs)
print(f"Valores únicos después de improve_t1: {np.unique(mask_improved)}")

unique, counts = np.unique(mask_improved, return_counts=True)
print("\nDistribución de clases en máscara mejorada:")
for u, c in zip(unique, counts):
    class_name = model.classes[u] if u < len(model.classes) else "unknown"
    percentage = c / mask_improved.size * 100
    print(f"  Clase {u} ({class_name}): {c} píxeles ({percentage:.2f}%)")

# Verificar si T1 está presente
t1_class = 1
if t1_class in unique:
    t1_count = counts[unique == t1_class][0]
    print(f"\n✅ T1 detectado: {t1_count} píxeles ({t1_count/mask_improved.size*100:.2f}%)")
else:
    print("\n⚠️  T1 aún no detectado. Puede ser necesario ajustar los umbrales.")

