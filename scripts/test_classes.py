"""Script para verificar las clases"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.model.segmentation_model import SegmentationModel

model = SegmentationModel('deeplab_hybrid')
model.load_model()
print('Clases cargadas:', model.classes)
print('Índice de F:', model._get_class_index('F'))
print('Índice de V:', model._get_class_index('V'))
print('Índice de T1:', model._get_class_index('T1'))
print('Índice de Background:', model._get_class_index('Background'))

