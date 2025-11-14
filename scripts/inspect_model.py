#!/usr/bin/env python3
"""
Script para inspeccionar la estructura del modelo guardado
"""
import torch
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentacion_app.app.config import MODEL_EXTRACTED_DIR


def inspect_model():
    """Inspecciona la estructura del modelo"""
    model_path = MODEL_EXTRACTED_DIR / "deeplabv3plus_best.pth"
    
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        return
    
    print("=" * 60)
    print("INSPECCIÓN DEL MODELO")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\nTipo del checkpoint: {type(checkpoint)}")
    print(f"\nKeys en el checkpoint: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nNúmero de parámetros en model_state_dict: {len(state_dict)}")
        print(f"\nPrimeras 10 keys del state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        
        print(f"\nÚltimas 10 keys del state_dict:")
        for i, key in enumerate(list(state_dict.keys())[-10:]):
            print(f"  {i+1}. {key}")
        
        # Analizar estructura
        print(f"\nAnálisis de la estructura:")
        keys_list = list(state_dict.keys())
        
        # Buscar patrones
        encoders = [k for k in keys_list if k.startswith('enc')]
        aspp_layers = [k for k in keys_list if 'aspp' in k]
        decoder_layers = [k for k in keys_list if 'decoder' in k]
        output_layers = [k for k in keys_list if k.startswith('out')]
        
        print(f"  - Capas encoder (enc*): {len(encoders)}")
        if encoders:
            print(f"    Ejemplo: {encoders[0]}")
        
        print(f"  - Capas ASPP (aspp*): {len(aspp_layers)}")
        if aspp_layers:
            print(f"    Ejemplo: {aspp_layers[0]}")
        
        print(f"  - Capas decoder (decoder*): {len(decoder_layers)}")
        if decoder_layers:
            print(f"    Ejemplo: {decoder_layers[0]}")
        
        print(f"  - Capas de salida (out*): {len(output_layers)}")
        if output_layers:
            print(f"    Ejemplo: {output_layers[0]}")
        
        # Verificar si es arquitectura personalizada
        first_key = keys_list[0] if keys_list else ""
        is_custom = 'enc1' in first_key or 'aspp' in first_key or 'decoder_conv' in first_key
        print(f"\n¿Arquitectura personalizada? {is_custom}")
        
        if is_custom:
            print("\n⚠️  Este modelo usa una arquitectura personalizada.")
            print("   Necesitas proporcionar la definición de la clase del modelo.")
            print("   Busca en tu código de entrenamiento donde defines:")
            print("   class TuModelo(nn.Module):")
            print("       def __init__(self, ...):")
            print("           ...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    inspect_model()

