"""
Subpaquete de modelos de inferencia para Express-Bound.

Contiene los módulos encargados de cargar los modelos ML desde Weights & Biases
y ejecutar inferencia sobre los datos en tiempo real:

- registry.py: dataclasses de entrada para cada modelo y clase ModelRegistry
  que gestiona la descarga y el registro de todos los modelos.
- dcrnn_infer.py: inferencia con el modelo DCRNN de propagación de retrasos.
- delay_infer.py: inferencia con los modelos LightGBM de retraso absoluto.
- delta_infer.py: inferencia con los modelos LightGBM de tendencia de retraso.
- alertas_infer.py: inferencia con el modelo XGBoost de alertas por línea.
"""
