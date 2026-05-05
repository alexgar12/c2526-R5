"""
Configuración centralizada de la aplicación Express-Bound.

Define la clase Settings, que agrupa todos los parámetros de configuración del
servidor de inferencia. Los valores se leen de variables de entorno, con valores
predeterminados razonables para desarrollo local.

Dependencias:
- Variables de entorno (os.getenv) para los nombres de artefactos W&B, rutas de
  tokens de Google Drive, y parámetros de inferencia.

Notas:
- En producción, los valores deben sobreescribirse mediante variables de entorno
  en el docker-compose o en el sistema de despliegue.
- La instancia global `settings` es importada por el resto de módulos de la app.
"""

import os
from pathlib import Path


class Settings:
    """
    Contenedor de configuración de la aplicación.

    Todos los atributos se resuelven en el momento de importación del módulo,
    leyendo las variables de entorno correspondientes o usando los valores por
    defecto indicados.
    """

    # Entidad y proyectos de Weights & Biases
    wandb_entity: str = os.getenv("WANDB_ENTITY", "pd1-c2526-team5")
    wandb_project_dcrnn: str = os.getenv("WANDB_PROJECT_DCRNN", "pd1-c2526-team5")
    wandb_project_delay: str = os.getenv("WANDB_PROJECT_DELAY", "pd1-c2526-team5")
    wandb_project_alertas: str = os.getenv("WANDB_PROJECT_ALERTAS", "pd1-c2526-team5")

    # Nombres de artefactos en W&B (deben coincidir con los scripts de entrenamiento)
    dcrnn_artifact: str = os.getenv("DCRNN_ARTIFACT", "dcrnn-final:latest")
    alertas_artifact: str = os.getenv("ALERTAS_ARTIFACT", "modelo_xgb_alertas:latest")
    lgbm_delay_30m_artifact: str = os.getenv("LGBM_DELAY_30M_ARTIFACT", "lgbm-delay-30m:latest")
    lgbm_delay_end_artifact: str = os.getenv("LGBM_DELAY_END_ARTIFACT", "lgbm-delay-end:latest")
    delta_10m_artifact: str = os.getenv("DELTA_10M_ARTIFACT", "lgbm-delta_delay_10m:latest")
    delta_20m_artifact: str = os.getenv("DELTA_20M_ARTIFACT", "lgbm-delta_delay_20m:latest")
    delta_30m_artifact: str = os.getenv("DELTA_30M_ARTIFACT", "lgbm-delta_delay_30m:latest")

    # Google Drive — el nombre de carpeta debe coincidir con upload_realtime_window.py
    google_drive_folder_name: str = os.getenv("GDRIVE_FOLDER_NAME", "MTA_Realtime_Windows")
    drive_token_path: Path = Path(
        os.getenv(
            "GDRIVE_TOKEN_PATH",
            str(
                Path(__file__).resolve().parent.parent
                / "src" / "ETL" / "alertas_oficiales_tiempo_real" / "token_drive.json"
            ),
        )
    )
    n_windows: int = int(os.getenv("N_WINDOWS", "12"))

    # Parámetros de inferencia
    data_cache_ttl: int = int(os.getenv("DATA_CACHE_TTL", "900"))
    alert_threshold: float = float(os.getenv("ALERT_THRESHOLD", "0.35"))


settings = Settings()
