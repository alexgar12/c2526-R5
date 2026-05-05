"""
Subpaquete de acceso y transformación de datos para Express-Bound.

Contiene los módulos encargados de obtener datos desde fuentes externas (Google Drive,
feeds GTFS-RT de la MTA) y de transformarlos en las estructuras que esperan los
modelos de inferencia.

Módulos incluidos:
- drive.py: descarga de ficheros y ventanas de datos desde Google Drive.
- train_infer.py: extracción de features por viaje desde el feed GTFS-RT.
- transforms.py: transformación de ventanas de datos en tensores o DataFrames
  listos para inferencia con DCRNN, LightGBM y XGBoost.
- vehicles.py: obtención de posiciones en tiempo real de los trenes desde los feeds MTA.
"""
