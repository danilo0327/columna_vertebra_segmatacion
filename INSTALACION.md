# Guía de Instalación

## ⚠️ Problema Común: Error `ModuleNotFoundError: No module named 'torch._C'`

Este error ocurre cuando PyTorch está instalado incorrectamente. Sigue estos pasos para solucionarlo.

## 🔧 Solución Rápida

### Opción 1: Usar el Script de Instalación (Recomendado)

**Windows:**
```powershell
.\install_dependencies.bat
```

**Linux/Mac:**
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### Opción 2: Instalación Manual

1. **Activar entorno virtual:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Desinstalar PyTorch existente (si está instalado):**
```bash
pip uninstall torch torchvision -y
```

3. **Instalar PyTorch desde el índice oficial:**

**Para CPU (recomendado para la mayoría):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Para GPU (si tienes CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Para GPU (si tienes CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. **Instalar el resto de dependencias:**
```bash
pip install -r segmentacion_app/requirements.txt
```

5. **Verificar instalación:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

## 📋 Instalación Completa desde Cero

### Windows

```powershell
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual
.\venv\Scripts\Activate.ps1

# 3. Actualizar pip
python -m pip install --upgrade pip

# 4. Instalar PyTorch primero (desde índice oficial)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Instalar resto de dependencias
pip install -r segmentacion_app/requirements.txt
```

### Linux/Mac

```bash
# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Actualizar pip
pip install --upgrade pip

# 4. Instalar PyTorch primero (desde índice oficial)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Instalar resto de dependencias
pip install -r segmentacion_app/requirements.txt
```

## ✅ Verificación

Después de la instalación, verifica que todo funciona:

```bash
python -c "import torch; import torchvision; import fastapi; import cv2; import numpy; print('Todas las dependencias instaladas correctamente')"
```

## 🐛 Solución de Problemas

### Error: "No module named 'torch._C'"

**Causa:** PyTorch instalado incorrectamente o incompleto.

**Solución:**
1. Desinstalar PyTorch: `pip uninstall torch torchvision -y`
2. Reinstalar desde índice oficial: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
3. Reiniciar el terminal/IDE

### Error: "numpy._core not found"

**Causa:** Incompatibilidad de versiones de numpy.

**Solución:**
```bash
pip uninstall numpy -y
pip install "numpy>=1.26.0,<2.0.0"
```

### Error al instalar PyTorch

**Solución:** Usa el índice oficial de PyTorch en lugar de PyPI:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 📝 Notas

- **PyTorch CPU vs GPU:** Para la mayoría de casos, la versión CPU es suficiente. Si tienes una GPU NVIDIA con CUDA, puedes instalar la versión GPU para mejor rendimiento.
- **Versión de Python:** Se recomienda Python 3.10 o 3.11.
- **Espacio en disco:** PyTorch requiere aproximadamente 2-3 GB de espacio.

## 🔗 Enlaces Útiles

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

