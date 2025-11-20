@echo off
REM Script de instalación de dependencias para Windows
echo ========================================
echo Instalando dependencias del proyecto
echo ========================================
echo.

REM Activar entorno virtual si existe
if exist venv\Scripts\activate.bat (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
) else (
    echo Creando entorno virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
)

echo.
echo Actualizando pip...
python -m pip install --upgrade pip

echo.
echo Instalando dependencias base (sin PyTorch)...
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6
pip install pillow==10.1.0
pip install "numpy>=1.26.0,<2.0.0"
pip install opencv-python==4.8.1.78
pip install pydantic==2.5.0
pip install "python-jose[cryptography]==3.3.0"
pip install pydicom==2.4.3

echo.
echo Instalando PyTorch (CPU) desde el índice oficial...
echo Esto puede tardar varios minutos...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Verificando instalación...
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
python -c "import fastapi; print('FastAPI instalado correctamente')"

echo.
echo ========================================
echo Instalación completada!
echo ========================================
pause

