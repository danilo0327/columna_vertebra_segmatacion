"""Script de diagnóstico para verificar por qué T1 no se segmenta"""
import sys
sys.path.insert(0, 'segmentacion_app')
from app.model.segmentation_model import SegmentationModel
from PIL import Image
import numpy as np
import torch

# Cargar modelo
print("Cargando modelo...")
model = SegmentationModel('deeplab_hybrid')
model.load_model()

# Crear imagen de prueba
test_img = Image.new('RGB', (512, 512), color='white')

# Preprocesar
input_tensor = model.preprocess_image(test_img)

# Predecir
print("Realizando predicción...")
with torch.no_grad():
    output = model.model(input_tensor)
    
    # Obtener probabilidades
    probs = torch.softmax(output, dim=1)
    probs_np = probs.squeeze(0).cpu().numpy()  # [C, H, W]
    
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO: Probabilidades por clase")
    print("=" * 60)
    for i in range(probs_np.shape[0]):
        class_name = model.classes[i] if i < len(model.classes) else f"Clase_{i}"
        prob_mean = probs_np[i].mean()
        prob_max = probs_np[i].max()
        prob_min = probs_np[i].min()
        prob_std = probs_np[i].std()
        pixels_above_03 = (probs_np[i] > 0.3).sum()
        pixels_above_05 = (probs_np[i] > 0.5).sum()
        total_pixels = probs_np[i].size
        
        print(f"\n{class_name} (Clase {i}):")
        print(f"  Probabilidad promedio: {prob_mean:.4f}")
        print(f"  Probabilidad máxima: {prob_max:.4f}")
        print(f"  Probabilidad mínima: {prob_min:.4f}")
        print(f"  Desviación estándar: {prob_std:.4f}")
        print(f"  Píxeles con prob > 0.3: {pixels_above_03} ({pixels_above_03/total_pixels*100:.2f}%)")
        print(f"  Píxeles con prob > 0.5: {pixels_above_05} ({pixels_above_05/total_pixels*100:.2f}%)")
    
    # Ver argmax
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    unique, counts = np.unique(pred_mask, return_counts=True)
    print("\n" + "=" * 60)
    print("Máscara después de argmax:")
    print("=" * 60)
    for u, c in zip(unique, counts):
        class_name = model.classes[u] if u < len(model.classes) else f"Clase_{u}"
        print(f"  {class_name} (Clase {u}): {c} píxeles ({c/pred_mask.size*100:.2f}%)")
    
    # Verificar si T1 tiene alguna probabilidad alta en alguna región
    t1_class = 1
    if t1_class < probs_np.shape[0]:
        t1_probs = probs_np[t1_class]
        print("\n" + "=" * 60)
        print("Análisis específico de T1:")
        print("=" * 60)
        print(f"  Probabilidad promedio T1: {t1_probs.mean():.4f}")
        print(f"  Probabilidad máxima T1: {t1_probs.max():.4f}")
        print(f"  Regiones con prob T1 > 0.2: {(t1_probs > 0.2).sum()} píxeles")
        print(f"  Regiones con prob T1 > 0.3: {(t1_probs > 0.3).sum()} píxeles")
        print(f"  Regiones con prob T1 > 0.4: {(t1_probs > 0.4).sum()} píxeles")
        
        # Ver qué clase gana donde T1 tiene alta probabilidad
        if (t1_probs > 0.2).any():
            high_t1_regions = t1_probs > 0.2
            winning_class = pred_mask[high_t1_regions]
            unique_win, counts_win = np.unique(winning_class, return_counts=True)
            print(f"\n  En regiones donde T1 tiene prob > 0.2, la clase ganadora es:")
            for uw, cw in zip(unique_win, counts_win):
                class_name = model.classes[uw] if uw < len(model.classes) else f"Clase_{uw}"
                print(f"    {class_name}: {cw} píxeles ({cw/len(winning_class)*100:.2f}%)")

