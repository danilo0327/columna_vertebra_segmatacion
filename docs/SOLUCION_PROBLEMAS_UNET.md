# Solución de Problemas: Error "Ran out of input" con modelos U-Net++

## 🔍 Diagnóstico del Problema

El error **"Ran out of input"** (EOFError) generalmente ocurre cuando:
1. El archivo del modelo no se descargó completamente desde Git LFS
2. El archivo está corrupto
3. Hay un problema de memoria al cargar el modelo

## ✅ Soluciones

### Solución 1: Verificar y Re-descargar desde Git LFS

Si clonaste el repositorio, los archivos grandes pueden no haberse descargado completamente:

```bash
# Verificar que Git LFS está instalado
git lfs version

# Si no está instalado, instálalo:
# Windows: Descarga desde https://git-lfs.github.com/
# Linux: sudo apt install git-lfs
# Mac: brew install git-lfs

# Inicializar Git LFS (si es la primera vez)
git lfs install

# Descargar todos los archivos LFS
git lfs pull

# Verificar que los archivos se descargaron
python scripts/test_model_loading.py
```

### Solución 2: Verificar Tamaño de Archivos

Los modelos U-Net++ deben tener aproximadamente **419 MB** cada uno:

```bash
# Windows PowerShell
Get-ChildItem models\unetplusplus\*.pth, models\unetplusplus_v2\*.pth | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}

# Linux/Mac
ls -lh models/unetplusplus/*.pth models/unetplusplus_v2/*.pth
```

Si el tamaño es menor (por ejemplo, solo unos KB), el archivo no se descargó correctamente.

### Solución 3: Re-descargar Manualmente

Si Git LFS no funciona, puedes descargar los modelos manualmente desde el repositorio:

1. Ve a tu repositorio en GitHub
2. Navega a `models/unetplusplus/unetplusplus_best.pth`
3. Haz clic en "Download" (GitHub debería mostrar un enlace de descarga para archivos LFS)
4. Reemplaza el archivo local

### Solución 4: Verificar Memoria Disponible

Los modelos U-Net++ son grandes y requieren memoria suficiente:

```bash
# Verificar memoria disponible
# Windows: Abre el Administrador de Tareas
# Linux: free -h
# Mac: Activity Monitor
```

Si tienes poca RAM (< 8 GB), considera:
- Cerrar otras aplicaciones
- Usar solo un modelo a la vez
- Reiniciar el servidor después de cada uso

### Solución 5: Probar Carga Manual

Ejecuta el script de prueba para verificar que los modelos se cargan correctamente:

```bash
python scripts/test_model_loading.py
```

Este script te dirá exactamente qué modelo tiene problemas.

## 🔧 Cambios Implementados

He mejorado el código para:
1. **Mejor manejo de errores**: Mensajes más descriptivos cuando falla la carga
2. **Carga en CPU primero**: Evita problemas de memoria con GPU
3. **Mejor logging**: Más información durante la carga del modelo
4. **Manejo de EOFError**: Mensajes específicos para este error

## 📝 Verificación

Después de aplicar las soluciones, verifica:

1. **Tamaño de archivos correcto**:
   - `unetplusplus_best.pth`: ~419 MB
   - `u_netplusplus_best.pth`: ~419 MB

2. **Carga exitosa**:
   ```bash
   python scripts/test_model_loading.py
   ```
   Debe mostrar "✅ OK" para todos los modelos

3. **Funcionamiento en la app**:
   - Reinicia el servidor
   - Intenta procesar una imagen con U-Net++
   - Debe funcionar sin errores

## 🆘 Si el Problema Persiste

Si después de seguir estos pasos el problema continúa:

1. **Verifica los logs del servidor**: Busca mensajes de error específicos
2. **Prueba con un modelo diferente**: Si DeepLabV3+ funciona pero U-Net++ no, el problema es específico de esos archivos
3. **Re-clona el repositorio**: A veces ayuda empezar desde cero
   ```bash
   git clone --recurse-submodules tu-repositorio
   cd columna_vertebra_segmatacion
   git lfs pull
   ```

## 📞 Información para Reportar el Error

Si necesitas ayuda adicional, proporciona:
- Tamaño de los archivos `.pth`
- Salida completa de `python scripts/test_model_loading.py`
- Mensaje de error completo del servidor
- Versión de Python y PyTorch

