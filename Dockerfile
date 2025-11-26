# Dockerfile para despliegue en AWS EC2
FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY segmentacion_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY segmentacion_app/ ./segmentacion_app/

# Copiar modelos (si existen)
COPY models/ ./models/

# Crear directorios necesarios
RUN mkdir -p models data segmentacion_app/app/static

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "segmentacion_app.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

