#!/usr/bin/env python3
"""
Script para ejecutar el servidor de desarrollo
"""
import uvicorn
from segmentacion_app.app.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    uvicorn.run(
        "segmentacion_app.app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )

