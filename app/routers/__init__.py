"""
Subpaquete de routers de la API FastAPI Express-Bound.

Contiene los módulos que definen los endpoints REST de la aplicación:

- health.py: endpoint GET /health con el estado del sistema (modelos y datos).
- predict.py: endpoints GET /api/predict/* para ejecutar los modelos ML y
  obtener predicciones de retraso, tendencia, propagación y alertas.
"""
