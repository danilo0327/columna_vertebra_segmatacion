#!/usr/bin/env python3
"""Script para probar la carga de modelos"""
import torch
import sys
from pathlib import Path

def test_model_loading(model_path):
    """Prueba cargar un modelo"""
    print(f"\n{'='*60}")
    print(f"Probando: {model_path}")
    print(f"{'='*60}")
    
    if not Path(model_path).exists():
        print(f"❌ ERROR: El archivo no existe: {model_path}")
        return False
    
    file_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"Tamaño del archivo: {file_size:.2f} MB")
    
    try:
        print("Intentando cargar el modelo...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Modelo cargado exitosamente")
        
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"Tipo: dict con {len(keys)} keys")
            print(f"Primeras 5 keys: {keys[:5]}")
            
            if 'model_state_dict' in checkpoint:
                print("✅ Tiene 'model_state_dict'")
                state_dict = checkpoint['model_state_dict']
                print(f"   Número de parámetros en state_dict: {len(state_dict)}")
                first_key = list(state_dict.keys())[0] if state_dict else None
                print(f"   Primera key: {first_key}")
            elif 'state_dict' in checkpoint:
                print("✅ Tiene 'state_dict'")
                state_dict = checkpoint['state_dict']
                print(f"   Número de parámetros en state_dict: {len(state_dict)}")
                first_key = list(state_dict.keys())[0] if state_dict else None
                print(f"   Primera key: {first_key}")
            elif 'model' in checkpoint:
                print("✅ Tiene 'model' (modelo completo)")
            else:
                # Verificar si las keys empiezan con "model."
                first_key = keys[0] if keys else ""
                if first_key.startswith("model."):
                    print("✅ Es un state_dict con prefijo 'model.'")
                else:
                    print(f"⚠️  Formato desconocido. Keys: {keys[:10]}")
        else:
            print(f"Tipo: {type(checkpoint)}")
            if hasattr(checkpoint, 'keys'):
                print(f"Tiene {len(list(checkpoint.keys()))} keys")
        
        return True
        
    except EOFError as e:
        print(f"❌ ERROR EOFError (Ran out of input): {e}")
        print("   Esto generalmente indica que el archivo está corrupto o incompleto.")
        print("   Posibles causas:")
        print("   - El archivo no se descargó completamente desde Git LFS")
        print("   - El archivo está corrupto")
        print("   - Problema de red durante la descarga")
        return False
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models_to_test = [
        "models/unetplusplus/unetplusplus_best.pth",
        "models/unetplusplus_v2/u_netplusplus_best.pth",
        "models/deeplabv3plus/deeplabv3plus_best.pth",
        "models/deeplab_resnet50/model_spine_t1_deeplabv3.pth"
    ]
    
    results = {}
    for model_path in models_to_test:
        results[model_path] = test_model_loading(model_path)
    
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    for model_path, success in results.items():
        status = "✅ OK" if success else "❌ ERROR"
        print(f"{status}: {model_path}")
    
    if not all(results.values()):
        print("\n⚠️  Algunos modelos tienen problemas. Verifica:")
        print("   1. Que los archivos se descargaron completamente desde Git LFS")
        print("   2. Ejecuta: git lfs pull")
        print("   3. Verifica el tamaño de los archivos")
        sys.exit(1)
    else:
        print("\n✅ Todos los modelos se cargan correctamente")

