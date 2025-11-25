#!/usr/bin/env python3
"""Script para analizar la estructura del modelo DeepLabV3pp"""
import torch
from collections import defaultdict

ckpt = torch.load('models/deeplab_hybrid/DeepLabV3pp_best.pth', map_location='cpu', weights_only=False)
keys = list(ckpt.keys())

# Agrupar por módulo
modules = defaultdict(list)
for key in keys:
    parts = key.split('.')
    if len(parts) > 0:
        modules[parts[0]].append(key)

print("="*60)
print("ANÁLISIS DE ESTRUCTURA DEL MODELO DeepLabV3pp")
print("="*60)

print(f"\nTotal de keys: {len(keys)}")
print(f"\nMódulos principales:")
for module in sorted(modules.keys()):
    print(f"  - {module}: {len(modules[module])} keys")

print("\n" + "="*60)
print("ESTRUCTURA DETALLADA POR MÓDULO")
print("="*60)

# Analizar ConvBlock (enc1)
print("\n1. ESTRUCTURA DE ConvBlock (enc1):")
enc1_keys = sorted([k for k in keys if k.startswith('enc1')])
conv_indices = set()
for k in enc1_keys:
    parts = k.split('.')
    if len(parts) >= 3:
        try:
            idx = int(parts[2])
            conv_indices.add(idx)
        except:
            pass
print(f"   Índices de convoluciones encontrados: {sorted(conv_indices)}")
for idx in sorted(conv_indices):
    idx_keys = [k for k in enc1_keys if k.split('.')[2] == str(idx)]
    print(f"   conv.{idx}: {len(idx_keys)} parámetros")
    for k in idx_keys[:3]:
        print(f"     - {k}: {ckpt[k].shape}")

# Analizar ASPP
print("\n2. ESTRUCTURA DE ASPP:")
aspp_keys = sorted([k for k in keys if 'aspp' in k.lower()])
print(f"   Total keys ASPP: {len(aspp_keys)}")
aspp_modules = defaultdict(list)
for k in aspp_keys:
    parts = k.split('.')
    if len(parts) >= 2:
        aspp_modules[parts[1]].append(k)
for module in sorted(aspp_modules.keys()):
    print(f"   {module}: {len(aspp_modules[module])} keys")
    for k in aspp_modules[module][:2]:
        print(f"     - {k}: {ckpt[k].shape}")

# Buscar decoder o salida
print("\n3. ESTRUCTURA DE DECODER/SALIDA:")
decoder_keys = [k for k in keys if 'decoder' in k.lower() or 'out' in k.lower() or 'head' in k.lower()]
if decoder_keys:
    print(f"   Total keys decoder: {len(decoder_keys)}")
    for k in decoder_keys[:10]:
        print(f"     - {k}: {ckpt[k].shape}")
else:
    print("   No se encontraron keys de decoder explícitas")
    # Buscar la última capa
    print("   Buscando capa de salida...")
    out_keys = [k for k in keys if 'out' in k or 'classifier' in k or 'final' in k]
    if out_keys:
        for k in out_keys:
            print(f"     - {k}: {ckpt[k].shape}")

# Mostrar todas las keys únicas de primer nivel
print("\n4. TODAS LAS KEYS DE PRIMER NIVEL:")
first_level = set([k.split('.')[0] for k in keys])
for fl in sorted(first_level):
    count = len([k for k in keys if k.startswith(fl)])
    print(f"   {fl}: {count} keys")

