# Proyecto PD1 - Grupo 5.

## Estructura de almacenamiento en MinIO

Los datos del proyecto se almacenan en un bucket S3-compatible (MinIO),
siguiendo una arquitectura tipo data lake organizada en distintas capas
según su nivel de procesamiento.

No se almacenan datos en GitHub.

Bucket utilizado: `pd1`
Raíz del proyecto: `grupo5/`

## Estructura general
```
pd1/
└── grupo5/
    ├── raw/
    │   ├── gtfs_static/    
    │   ├── gtfs_realtime/
    │   ├── weather/
    │   ├── official_alerts/
    │   └── events/
    │
    ├── processed/
    │   ├── gtfs_static/
    │   ├── gtfs_realtime/
    │   ├── weather/
    │   ├── official_alerts/
    │   └── events/
    │
    ├── cleaned/
    │   ├── gtfs_clean/
    │   ├── weather_clean/
    │   ├── events_clean/
    │   ├── official_alerts_clean/
    │   └── quality_reports/      → métricas de calidad de datos
    │
    └── analytics/
        ├── features_dataset/     → dataset final
        ├── headway_analysis/     → análisis de desviaciones
        ├── delay_analysis/       → estadísticas de retraso
        ├── anomaly_labels/       → etiquetas binarias para modelado
        └── snapshots_realtime/   → agregaciones en tiempo real
```
## Descripción de cada capa

### raw/
Contiene los datos originales descargados de las fuentes externas
(GTFS histórico, GTFS-Realtime, datos meteorológicos, eventos, avisos).
No se modifican una vez almacenados.

### processed/
Datos transformados a un formato estructurado (principalmente Parquet),
pero todavía sin limpieza exhaustiva.

### cleaned/
Datos limpios y validados. Incluye:
- Eliminación de duplicados
- Corrección de tipos
- Control de outliers
- Reportes de calidad

También contiene features que no impliquen agregaciones temporales (p.ej. delay_sec, para agregar en el dataset final).

### analytics/
Conjunto final de datos listos para análisis y modelado.
Incluye el dataset con features agregados.

## Convención de nombres
Los objetos se almacenan siguiendo la convención:

grupo5/raw/gtfs_static/date=YYYY-MM-DD/nombre_archivo.parquet

Lo cual permite:
- Filtrado eficiente por fecha
- Procesamiento incremental

