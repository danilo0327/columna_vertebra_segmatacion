# Aplicación principal
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from .config import APP_NAME, APP_VERSION, STATIC_DIR, TEMPLATES_DIR
from .api import router

# Crear instancia de la aplicación
_app = None


def get_app() -> FastAPI:
    """Obtiene o crea la instancia de la aplicación FastAPI"""
    global _app
    if _app is None:
        _app = FastAPI(
            title=APP_NAME,
            version=APP_VERSION,
            description="API para segmentación de columna vertebral y vértebra T1 en radiografías"
        )
        
        # Incluir rutas
        _app.include_router(router)
        
        # Montar archivos estáticos
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        _app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    return _app


# Crear la aplicación al importar
app = get_app()


@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación"""
    print(f"Iniciando {APP_NAME} v{APP_VERSION}")
    print(f"Directorio de modelos: {Path(__file__).parent.parent.parent / 'models'}")
    print(f"Directorio de estáticos: {STATIC_DIR}")

