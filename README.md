<img width="512" height="512" alt="Logo1PD" src="https://github.com/user-attachments/assets/44ea9a4a-ce36-4497-9ec0-779366090aa4" />

# Express-Bound
Proyecto de Datos I – Grupo 5

Facultad de Informática – UCM

## Descripción del proyecto

Express-Bound integra datos operativos y contextuales del metro de Nueva York para estimar retrasos a corto plazo y anticipar incidencias.

El proyecto se centra en tres líneas principales:

1. Predicción del retraso de un tren concreto.
2. Modelado de la propagación de los retrasos por la red.
3. Detección temprana de incidencias operativas.

El enfoque es de predicción a corto horizonte (10–30 minutos), utilizando tanto el estado actual de la red como información contextual (clima, calendario, estructura de la red).

El sistema está diseñado siguiendo una arquitectura tipo data lake (raw → processed → cleaned) sobre almacenamiento en MinIO, garantizando trazabilidad y reproducibilidad del pipeline.

## Estructura del proyecto
```
├── src/
│   ├── common/                        # Utilidades compartidas (MinIO client, etc.)
│   ├── ETL/                           # Scripts de ingestión, limpieza y generación de features
│   │   ├── alertas_oficiales_tiempo_real/   # Alertas MTA (histórico y tiempo real)
│   │   ├── clima/                     # Datos meteorológicos (Open-Meteo)
│   │   ├── eventos/                   # Eventos NYC (deportes, conciertos, oficiales)
│   │   ├── gtfs_historico/            # GTFS histórico (instancias de trenes pasando por estaciones)
│   │   ├── pipelines/
│   │   │   ├── historical/            # Orquestadores y scripts del pipeline batch (extracción, transformación, dataset final)
│   │   │   └── realtime/              # Worker y scripts del pipeline en tiempo real (inferencia, ventanas, subida a Drive)
│   │   └── tiempo_real_metro/         # GTFS en tiempo real (MTA feeds)
│   └── models/                        
│       ├── common/                    # Agregaciones temporales adicionales
│       ├── modelos_alertas/           # Modelos entrenados para anticipar las alertas oficiales de la MTA
│       ├── prediccion_retrasos/       # Modelos entrenados para predecir el retraso de trenes
│       ├── propagacion_estacion/      # Modelos entrenados para modelar la propagación del retraso por la red 
│       └── seleccion_variables.md     # Explicación de las variables que mantenemos en la fase de modelado a partir de los resultados de los notebooks de análisis
│
├── app/                               # API REST y cliente web (FastAPI)
│   ├── app.py                         # Punto de entrada de la aplicación
│   ├── config.py                      # Configuración y variables de entorno
│   ├── cache.py                       # Caché en memoria para datos de inferencia
│   ├── schemas.py                     # Esquemas Pydantic de request/response
│   ├── routers/
│   │   ├── predict.py                 # Endpoints de predicción
│   │   └── health.py                  # Endpoint de health check
│   ├── models/
│   │   ├── registry.py                # Carga y registro de modelos
│   │   ├── delay_infer.py             # Inferencia delay_30m y delay_end
│   │   ├── delta_infer.py             # Inferencia delta_delay
│   │   ├── alertas_infer.py           # Inferencia anticipación de alertas
│   │   └── dcrnn_infer.py             # Inferencia propagación (DCRNN)
│   ├── data/
│   │   ├── drive.py                   # Lectura de datos desde MinIO/Drive
│   │   └── transforms.py              # Transformaciones previas a la inferencia
│   ├── static/                        # CSS y JS del cliente web
│   └── templates/                     # Plantillas HTML del cliente web
│
├── notebooks/                         # Análisis exploratorio y visualizaciones
├── docs/                              # Documentación adicional del proyecto
├── Dockerfile                         # Imagen del contenedor (API + worker RT)
├── .dockerignore                      # Excluye notebooks, artefactos, .env y cachés del contexto de build
├── pyproject.toml                     # Dependencias y configuración del proyecto (uv)
├── uv.lock                            # Lockfile de dependencias para builds reproducibles
├── .env.example                       # Plantilla de variables de entorno
├── .gitignore                         # Ficheros excluidos del control de versiones
└── README.md                          # Este fichero
```

## Almacenamiento en MinIO

Los datos del proyecto se almacenan en un bucket S3-compatible (MinIO),
siguiendo una arquitectura tipo data lake organizada en distintas capas
según su nivel de procesamiento.

Bucket utilizado: `pd1`
Raíz del proyecto: `grupo5/`

### Estructura 

```
pd1/
└── grupo5/
    ├── raw/
    │   ├── avisos_oficiales_historico_2025/
    │   └── eventos_nyc/
    │
    ├── processed/
    │   ├── eventos_nyc/
    │   ├── gtfs_realtime/
    │   ├── gtfs_with_delays/
    │   ├── clima/
    │   └── official_alerts/
    │
    ├── final/
    │   ├── year=2025/
    │   │   └── month=*/
    │   └── year=2026/
    │       └── month=*/
    │
    ├── aggregations/
    │   └── lines/
    │
    ├── cleaned/
    │   ├── clima_clean/
    │   ├── eventos_nyc/
    │   ├── gtfs_clean_scheduled/
    │   ├── gtfs_clean_unscheduled/
    │   └── official_alerts/
    │
    └── realtime/
```
## Descripción de cada capa

### raw/
Contiene los datos originales descargados de las fuentes externas y sin tratar.
No se modifican una vez almacenados.

### processed/
Datos transformados a un formato estructurado (principalmente Parquet),
unidos de distintas fuentes pero todavía sin limpieza exhaustiva.

### cleaned/
Datos limpios y validados. Incluye:
- Eliminación de duplicados
- Corrección de tipos
- Control de outliers
- Reportes de calidad


### final/
Dataset final con todas las fuentes integradas y listas para el modelado. Los datos se organizan por año (`year=2025/`, `year=2026/`) y dentro de cada año por mes (`month=*/`), un Parquet por mes.

### aggregations/
Dataset completo de 2025 y 2026 con todos los meses agregados a resolución de 60 minutos. Sirve como entrada para los modelos de propagación y análisis a nivel de red, y de alertas.

### realtime/
Estado actual de la red almacenado por el worker de tiempo real. Se sobreescribe de forma continua con los datos más recientes procedentes de los feeds GTFS-RT de la MTA.

## Convención de nombres
Los objetos se almacenan siguiendo la convención:

grupo5/processed/nombre_fuente/date=YYYY-MM-DD/nombre_archivo.parquet

Lo cual permite:
- Filtrado eficiente por fecha
- Procesamiento incremental
- Re-ejecución parcial del pipeline en caso de fallo

## Configuración del entorno de desarrollo

El proyecto utiliza Python y el gestor de dependencias `uv`.

### Requisitos previos

- Python >= 3.13
- uv instalado
- Acceso a MinIO (credenciales proporcionadas al grupo)

### Configuración de variables de entorno

Se recomienda crear un fichero `.env` en la raíz del proyecto (se puede usar `.env.example` como plantilla). Este fichero es utilizado tanto por los scripts locales como por el contenedor Docker (`--env-file .env`).

```
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MOBILITY_DATABASE_REFRESH_TOKEN=...
NYC_OPEN_DATA_TOKEN=...
CLIENT_ID_SEATGEEK=...
SETLIST_API_KEY=...
WANDB_API_KEY=...
```

Además se requieren las credenciales de Gmail para la ingestión de alertas oficiales:
```
Gmail credentials
Gmail token
```

### Token de Google Drive (producción)

La API necesita un token OAuth2 en formato JSON para acceder a Google Drive en producción. Genera el token localmente con el script proporcionado:

```bash
uv run python generar_token_drive.py
```

El token resultante se guarda por defecto en `src/ETL/alertas_oficiales_tiempo_real/token_drive.json`.

Puedes añadir directamente en tu fichero `.env` la línea de ejemplo:

```bash
GDRIVE_TOKEN_JSON="$(cat src/ETL/alertas_oficiales_tiempo_real/token_drive.json)"
```

La aplicación leerá la variable `GDRIVE_TOKEN_JSON` con el JSON completo del token y podrá parsearla en runtime (si es necesario, se puede escribir a `src/ETL/alertas_oficiales_tiempo_real/token_drive.json` para compatibilidad con las librerías de Google).

Para instrucciones sobre cómo ejecutar el contenedor y pasar variables/volúmenes, consulta la sección **Despliegue con Docker** al final de este README.

Notas:
- Asegúrate de tener `credentials.json` en la raíz del proyecto antes de ejecutar `generar_token_drive.py`.
- Evita subir `token_drive.json` a repositorios públicos; gestiona el secreto mediante variables de entorno o un gestor de secretos en CI/CD.

### Weights & Biases (W&B)

El proyecto utiliza [Weights & Biases](https://wandb.ai) para el seguimiento de experimentos de todos los modelos. Cada entrenamiento registra automáticamente métricas, hiperparámetros y artefactos. Para activarlo es necesario proporcionar `WANDB_API_KEY` en el `.env`. Los runs se pueden consultar en el proyecto del equipo en la plataforma de W&B.

### Crear entorno e instalar dependencias

```bash
uv sync
```

## Ejecución de los pipelines

El proyecto está automatizado mediante dos orquestadores principales ubicados en:

```
src/ETL/pipelines/
```

Estos permiten ejecutar la ingesta y transformación de datos de forma parametrizable y reproducible.

---

### Extracción de datos

Script principal:

```
src/ETL/pipelines/historical/run_extraccion.py
```

Este orquestador ejecuta la descarga de datos desde las distintas fuentes externas (GTFS, clima, eventos, alertas oficiales, etc.) y los almacena en la capa `raw/` de MinIO.

#### Parámetros disponibles

- `--source`: nombre de la fuente específica o `all`
- `--start`: fecha de inicio (formato YYYY-MM-DD)
- `--end`: fecha de fin (formato YYYY-MM-DD)

#### Ejemplo de ejecución

```bash
uv run python src/ETL/pipelines/historical/run_extraccion.py --source all --start 2025-01-01 --end 2025-01-03
```

---

### Transformación de datos

Script principal:

```
src/ETL/pipelines/historical/run_transform.py
```

Este orquestador procesa los datos almacenados en `raw/` y/o `processed/`, realiza limpieza, integración y generación de variables, y los mueve a capas superiores del data lake.

#### Parámetros disponibles

- `--source`
- `--start`
- `--end`
- `--continue_on_error`

#### Ejemplo de ejecución

```bash
uv run python src/ETL/pipelines/historical/run_transform.py --source all --start 2025-01-01 --end 2025-01-03
```

Tras su ejecución, los datos seguirán el flujo:

```
raw/ → processed/ → cleaned/
```

---

### Construcción del dataset final

Una vez completada la transformación, tres scripts adicionales preparan los datos para el modelado:

```bash
# Une todas las fuentes limpias en un único parquet mensual con todos los features (cleaned/ → final/)
uv run python src/ETL/pipelines/historical/generate_final_dataset.py --start 2025-01-01 --end 2025-01-31

# Divide el dataset final por línea de metro (necesario para modelos por línea)
uv run python src/ETL/pipelines/historical/split_final_by_line.py

# Agrega los parquets mensuales de cada línea en un único parquet anual
uv run python src/ETL/pipelines/historical/aggregate_lines_yearly.py
```

---

### Flujo completo

```
raw/ → processed/ → cleaned/ → final/ → aggregations/
```

---

## Arquitectura interna del pipeline

### Pipeline histórico

Los dos orquestadores principales (`run_extraccion`, `run_transform`) utilizan un registro de fuentes (`REGISTRY`) que mapea cada nombre de fuente a su función correspondiente. Esto permite ejecutar una sola fuente o todas con el mismo comando. Flujo completo:

```
run_extraccion          →  raw/               (descarga de 4 fuentes: GTFS histórico, clima, eventos, alertas)
run_transform           →  processed/ → cleaned/   (limpieza + generación de features, por fuente)
generate_final_dataset  →  final/             (une todas las fuentes limpias en un Parquet mensual)
split_final_by_line     →  final/ por línea
aggregate_lines_yearly  →  aggregations/      (Parquet anual agregado a resolución de 60 min)
```

### Pipeline en tiempo real

- **`generate_realtime_dataset.py`** pieza central del pipeline. Fusiona en tiempo real las cuatro fuentes de datos (GTFS-RT, Open-Meteo, SeatGeek/ESPN/NYC Open Data y alertas Gmail MTA) y produce un dataframe con el mismo esquema de columnas que el dataset histórico, garantizando compatibilidad directa con los modelos entrenados.

- **`aggregate_realtime_dataset.py`** equivalente en tiempo real de las agregaciones históricas. Recibe el dataframe de `generate_realtime_dataset` y lo agrega en ventanas temporales de X minutos (por defecto 30), agrupando por parada, línea y dirección.

- **`upload_realtime_window.py`** orquesta la ejecución del pipeline RT completo y sube la ventana agregada más reciente a Google Drive, manteniendo solo las N ventanas más recientes. La carpeta la gestiona una cuenta de servicio y se comparte automáticamente con los emails configurados en `GDRIVE_SHARE_EMAILS`.

- **`upload_daily_data.py`** se ejecuta una vez al día a las 00:00 hora NY (cron en VM de Google Cloud). Sube a Google Drive los datos estáticos del día: GTFS `stop_times`, clima y eventos. Esto evita tener que hacer llamadas a las apis repetidamente obteniendo el mismo resultado.

- **`preprocess_realtime_lgbm.py`** construye el vector de features de inferencia para un `match_key` concreto, necesario para los modelos `LightGBM`. Lee los datos estáticos desde Google Drive (actualizados por `upload_daily_data.py` para agilizar el proceso) y hace las únicas dos llamadas en tiempo real necesarias: GTFS-RT de la línea del viaje y alertas Gmail MTA. También gestiona el estado de lags (delays pasados) en MinIO para poder conseguir el delay de un tren en sus dos paradas anteriores, necesarios para predecir.

- **`local_realtime_worker.py`** proceso continuo que se ejecuta dentro del contenedor (cada 90s). Llama a `preprocess_realtime_lgbm.update_lag_state()` para mantener el estado de lags de todos los viajes activos en MinIO. De este modo la API siempre tiene el retraso de las paradas anteriores de un tren actualizadas.

---

## Uso de notebooks

Los notebooks del proyecto están ubicados en:

```
notebooks/
```

Se utilizan para:

- Análisis exploratorio de datos (EDA)
- Validación de variables derivadas
- Visualización de resultados


---
## Uso de modelos de ML

Todos los modelos se almacenan en la carpeta models, la cual está dividida por problemas:

### 1. Modelos de anticipación de alertas

Se almacenan en la carpeta

```
modelos_alertas/
```

En ella encontramos los modelos de Regresión Logística, Random Forest y XGBoost, cuyos hiperparámetros se han obtenido de búsquedas con Optuna y Random Search
almacenados en:

```
optuna/
random/
```

Los entrenamientos y evaluación se almacenan en 

```
common/
```

El análisis de desempeño de los nuevos datos se almacena en 

```
analytics/
```

### 2. Modelos de predicción de retrasos a nivel de tren

Se almacenan en la carpeta

```
prediccion_retrasos/
```

Este modelo se divide en 4 subproblemas

- Predicción de retrasos a 30 minutos vista para los trenes cuyo que estarán en funcionamiento más de 30 minutos.

```
delay_30m/
```

- Predicción del retraso final para trenes que acabarán su viaje antes de 30 minutos.

```
delay_end/
```
- Predicción de comportamiento del retraso (mejora o empeora).

```
delta/
```

- Predicción del retraso por intervalos.

```
prediccion_por_intervalos/
```

Todos ellos a su vez se organizan en carpetas análogas a las anteriores.

```
optuna/
random/
test/
```
Los análisis de desempeño con los nuevos datos se almacenan en

```
analytics/
```

### 3. Modelos de propagación de retrasos

Se almacenan en la carpeta

```
propagacion_estacion/
```

Modela cómo un retraso en una estación se propaga por la red de metro usando *Graph Neural Networks* (GNN). La red está representada como un grafo de 899 nodos (paradas en estaciones) con pesos calculados mediante Gaussian Kernel sobre los tiempos medianos de viaje entre ellas.

Se implementan y comparan tres arquitecturas GNN:

- **DCRNN** (*Diffusion Convolutional Recurrent Neural Network*): combina convolución difusiva sobre el grafo con GRU para capturar dependencias espaciales y temporales.
- **STGCN** (*Spatio-Temporal Graph Convolutional Network*): bloques de convolución espacial y temporal en paralelo.
- **ASTGCN** (*Attention-based STGCN*): añade mecanismos de atención espacial y temporal sobre STGCN.

#### Scripts del pipeline (ejecución en orden)

| Script | Descripción |
|--------|-------------|
| `01_generar_grafo.py` | Procesa el GTFS histórico desde MinIO y construye el grafo de la red: `edge_index`, `edge_weight` y lista de nodos. Guarda `artefactos/grafo.pt`. |
| `02_generar_tensores.py` | Genera los tensores espacio-temporales X e Y a partir del dataset final, aplica el split cronológico train/val/test y escala con `StandardScaler`. Guarda `artefactos/tensores.pt`. |
| `03_baseline_naive.py` | Baseline 1 — predice para todos los horizontes el último valor observado de retraso. Sirve de cota inferior de referencia. |
| `04_baseline_ha.py` | Baseline 2 — Historical Average. Calcula la media de retraso histórica por (estación, día de semana, hora) en train y la aplica como predicción en test. Captura patrones cíclicos sin aprendizaje. |
| `05_ablacion_features.py` | Entrena variantes de DCRNN con subconjuntos de features (retraso base, contexto, calendario) durante pocas épocas para identificar el conjunto óptimo. Guarda `artefactos/ablacion.pt`. |
| `06_tuning_hpo_dcrnn.py` | HPO de DCRNN con Optuna y Random Search. Guarda el mejor conjunto de hiperparámetros en `artefactos/hpo.pt`. |
| `07_tuning_stgcn.py` | HPO de STGCN con Optuna. Guarda en `artefactos/stgcn_hpo.pt`. |
| `08_tuning_astgcn.py` | HPO de ASTGCN con Optuna. Guarda en `artefactos/astgcn_hpo.pt`. |
| `09_entrenamiento_final_dcrnn.py` | Entrena DCRNN con los mejores hiperparámetros cargando los artefactos pre-calculados. Guarda `artefactos/dcrnn_final.pth`. |
| `10_entrenamiento_final_stgcn.py` | Entrena STGCN final. Guarda `artefactos/stgcn_final.pth`. |
| `11_entrenamiento_final_astgcn.py` | Entrena ASTGCN final. Guarda `artefactos/astgcn_final.pth`. |
| `12_evaluacion_modelos.py` | **Único script que toca el conjunto de test.** Evaluación comparativa de los tres modelos: métricas principales (MAE, RMSE, R²), análisis por segmentos (fin de semana, clima extremo) y Permutation Feature Importance. |

#### Módulos de soporte

```
models/        — implementaciones de DCRNN, STGCN y ASTGCN
utils/         — dataset.py (carga y ventanas deslizantes), metrics.py (MAE, RMSE, R²)
artefactos/    — artefactos generados por los scripts anteriores
```

---

## Resultados de los mejores modelos

### 1. Anticipación de alertas

El modelo ganador es **XGBoost**, con un PR-AUC de **0.795** sobre el conjunto de test (baseline aleatorio: 0.145).

Distribuye el peso predictivo entre múltiples variables, siendo `seg_desde_ultima_alerta_linea` la más relevante (importancia ~0.37), a diferencia de Random Forest y Regresión Logística que dependen de ella de forma excesiva (~0.73).

| Modelo                | PR-AUC |
|----------------------|-------:|
| Baseline aleatorio   | 0.145  |
| Regresión Logística  | 0.44   |
| Random Forest        | 0.755  |
| **XGBoost**          | **0.795** |

---

### 2. Predicción de retrasos a nivel de tren

Los modelos seleccionados para producción son:

- **LightGBM (regresión)** → `delay_30m` y `delay_end`
- **LightGBM (clasificación binaria)** → comportamiento del retraso (`delta_delay`)

Los modelos por intervalos se descartaron por su escasa mejora respecto al baseline.

#### Regresión por horizonte temporal

| Modelo                 | MAE delay_30m (s) | MAE delay_end (s) | R² delay_end |
|-----------------------|------------------:|------------------:|-------------:|
| Baseline media        | 220               | 232               | ~0           |
| Baseline persistencia | 160               | 156               | 0.453        |
| **LightGBM**          | **134**           | **109**           | **0.696**    |
| MLP                   | 137               | 121               | 0.686        |

#### Clasificación binaria del `delta_delay` (media entre horizontes 10–60 min)

| Modelo               | ROC-AUC |
|---------------------|--------:|
| Baseline mayoritario| 0.50    |
| Random Forest       | 0.801   |
| **LightGBM**        | **0.828** |

Las variables más influyentes en todos los modelos son:

- `delay_seconds` (retraso actual)  
- `lagged_delay_1` (retraso retardado)  
- `delay_velocity` (velocidad de cambio del retraso)

---

### 3. Propagación de retrasos a nivel de estación (GNN)

El modelo ganador es **DCRNN**, con un MAE global de **81.24 segundos** en test, superando a STGCN y ASTGCN y a los dos baselines.

Destaca especialmente durante los fines de semana (**MAE de 50.58 s**).

| Modelo                  | MAE (s) | RMSE (s) | R²     |
|--------------------------|--------:|----------:|-------:|
| Baseline media           | 102.27  | 185.32    | 0.051  |
| Baseline persistencia    | 105.79  | 225.00    | -0.400 |
| STGCN                    | 95.08   | 189.09    | 0.021  |
| ASTGCN                   | 93.66   | 175.83    | 0.151  |
| **DCRNN**                | **81.24** | 177.60  | 0.134  |

El análisis de importancia (*Permutation Feature Importance*) confirma que las variables más relevantes son:

- `delay_seconds` (retraso actual)  
- hora del día  
- `route_rolling_delay` (congestión de red)

---

## Despliegue con Docker

El proyecto incluye un `Dockerfile` que empaqueta tanto la API REST como el worker de ingestión en tiempo real en un único contenedor.

### Construcción de la imagen

```bash
docker build -t express-bound .
```

### Ejecución del contenedor

```bash
docker run -p 8000:8000 --env-file .env express-bound
```

El fichero `.env` debe contener las variables de entorno descritas en la sección [Configuración del entorno de desarrollo](#configuración-del-entorno-de-desarrollo).

### Qué arranca el contenedor

Al iniciarse, el contenedor lanza dos procesos en paralelo:

- **API REST** (`app/app.py`) — servidor FastAPI accesible en `http://localhost:8000`. Expone los endpoints de predicción que consumen los modelos entrenados.
- **Worker de tiempo real** (`src/ETL/pipelines/realtime/local_realtime_worker.py`) — proceso en segundo plano que se conecta a los feeds GTFS-RT de la MTA, descarga el estado actual de la red de metro y lo procesa de forma continua para tenerlo disponible para la inferencia. Sin este worker, la API no dispone de datos frescos con los que generar predicciones.

---

## Autores

| | | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [<img src="https://github.com/34maario.png" width="100" height="100">](https://github.com/34maario) | [<img src="https://github.com/alexgar12.png" width="100" height="100">](https://github.com/alexgar12) | [<img src="https://github.com/chiaralg06.png" width="100" height="100">](https://github.com/chiaralg06) | [<img src="https://github.com/davidr210.png" width="100" height="100">](https://github.com/davidr210) | [<img src="https://github.com/IvanGavaaaa.png" width="100" height="100">](https://github.com/IvanGavaaaa) | [<img src="https://github.com/juannjurado.png" width="100" height="100">](https://github.com/juannjurado) | [<img src="https://github.com/juliahuergoucm.png" width="100" height="100">](https://github.com/juliahuergoucm) | [<img src="https://github.com/sergioduenas10.png" width="100" height="100">](https://github.com/sergioduenas10) |
| [Mario González](https://github.com/34maario) | [Alex García](https://github.com/alexgar12) | [Chiara Gómez](https://github.com/chiaralg06) | [David Rodríguez](https://github.com/davidr210) | [Iván García](https://github.com/IvanGavaaaa) | [Juan Jurado](https://github.com/juannjurado) | [Julia Huergo](https://github.com/juliahuergoucm) | [Sergio Dueñas](https://github.com/sergioduenas10) |

Curso 2025-2026
