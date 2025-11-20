#!/bin/bash
# Script de instalación de dependencias para Linux/Mac

echo "========================================"
echo "Instalando dependencias del proyecto"
echo "========================================"
echo ""

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
else
    echo "Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo ""
echo "Actualizando pip..."
pip install --upgrade pip

echo ""
echo "Instalando dependencias base (sin PyTorch)..."
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6
pip install pillow==10.1.0
pip install "numpy>=1.26.0,<2.0.0"
pip install opencv-python==4.8.1.78
pip install pydantic==2.5.0
pip install "python-jose[cryptography]==3.3.0"
pip install pydicom==2.4.3

echo ""
echo "Instalando PyTorch (CPU) desde el índice oficial..."
echo "Esto puede tardar varios minutos..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Verificando instalación..."
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
python3 -c "import fastapi; print('FastAPI instalado correctamente')"

echo ""
echo "========================================"
echo "Instalación completada!"
echo "========================================"

