"""Script para probar el nuevo modelo deeplab_dense_decoder"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.config import AVAILABLE_MODELS
from app.model.segmentation_model import SegmentationModel

print("=" * 60)
print("Verificando configuración del nuevo modelo")
print("=" * 60)

# Verificar que el modelo está en la configuración
if "deeplab_dense_decoder" in AVAILABLE_MODELS:
    model_config = AVAILABLE_MODELS["deeplab_dense_decoder"]
    print(f"✓ Modelo encontrado en configuración:")
    print(f"  Nombre: {model_config['name']}")
    print(f"  Directorio: {model_config['model_dir']}")
    print(f"  Archivo: {model_config['model_file']}")
    print(f"  Clases: {model_config['classes_file']}")
    print(f"  Arquitectura: {model_config['architecture']}")
    
    # Verificar que los archivos existen
    import os
    model_path = model_config['model_dir'] / model_config['model_file']
    classes_path = model_config['model_dir'] / model_config['classes_file']
    
    print(f"\n✓ Verificando archivos:")
    print(f"  Modelo: {model_path.exists()} - {model_path}")
    print(f"  Clases: {classes_path.exists()} - {classes_path}")
    
    if model_path.exists() and classes_path.exists():
        print("\n✓ Intentando cargar el modelo...")
        try:
            model = SegmentationModel('deeplab_dense_decoder')
            model.load_model()
            print("✓ Modelo cargado exitosamente!")
            print(f"  Clases: {model.classes}")
            print(f"  Dispositivo: {model.device}")
        except Exception as e:
            print(f"✗ Error al cargar el modelo: {e}")
    else:
        print("✗ Algunos archivos no existen")
else:
    print("✗ Modelo 'deeplab_dense_decoder' no encontrado en configuración")

print("\n" + "=" * 60)

