#!/usr/bin/env python3
"""
Script para extraer y preparar el modelo desde el archivo ZIP
"""
import zipfile
import json
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentacion_app.app.config import MODEL_ZIP_PATH, CLASSES_JSON_PATH, MODEL_EXTRACTED_DIR


def extract_model():
    """Extrae el modelo del archivo ZIP"""
    print(f"Extrayendo modelo desde: {MODEL_ZIP_PATH}")
    
    if not MODEL_ZIP_PATH.exists():
        print(f"Error: No se encontró el archivo {MODEL_ZIP_PATH}")
        return False
    
    # Crear directorio de destino
    MODEL_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_EXTRACTED_DIR)
        print(f"Modelo extraído exitosamente en: {MODEL_EXTRACTED_DIR}")
        
        # Listar archivos extraídos
        print("\nArchivos extraídos:")
        for file_path in MODEL_EXTRACTED_DIR.rglob('*'):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(MODEL_EXTRACTED_DIR)}")
        
        return True
    except Exception as e:
        print(f"Error al extraer el modelo: {e}")
        return False


def verify_classes():
    """Verifica que el archivo de clases existe y es válido"""
    print(f"\nVerificando clases desde: {CLASSES_JSON_PATH}")
    
    if not CLASSES_JSON_PATH.exists():
        print(f"Error: No se encontró el archivo {CLASSES_JSON_PATH}")
        return False
    
    try:
        with open(CLASSES_JSON_PATH, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        print(f"Clases cargadas: {classes}")
        return True
    except Exception as e:
        print(f"Error al cargar clases: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Script de Extracción de Modelo")
    print("=" * 50)
    
    success = True
    success &= extract_model()
    success &= verify_classes()
    
    if success:
        print("\n✅ Preparación completada exitosamente")
    else:
        print("\n❌ Hubo errores en la preparación")
        sys.exit(1)

