"""
Punto de entrada principal de la API FastAPI Express-Bound.

Define la aplicación FastAPI, gestiona el ciclo de vida del servidor (arranque y
parada), carga todos los modelos desde Weights & Biases, inicializa las cachés
TTL, descarga los datos estáticos de GTFS y expone los endpoints REST y WebSocket
de la aplicación.

Dependencias principales:
- FastAPI para el servidor HTTP y WebSocket.
- app.cache.TTLCache para cachear ventanas de datos y posiciones de vehículos.
- app.config.settings para la configuración centralizada vía variables de entorno.
- app.models.registry.ModelRegistry para la carga de modelos ML.
- app.data.vehicles.fetch_positions para obtener posiciones de trenes en tiempo real.
- Google Drive (app.data.drive) para metadatos de estaciones y datos estáticos GTFS.

Notas:
- Los modelos se cargan de forma concurrente al arrancar el servidor.
- Los datos GTFS estáticos se descargan desde la URL pública de la MTA.
- El endpoint WebSocket /ws/live-updates emite alertas cada 60 segundos.
"""

import asyncio
import io
import json
import logging
import os
import random
import urllib.request
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware

from app.cache import TTLCache
from app.config import settings
from app.data.vehicles import fetch_positions
from app.models.registry import ModelRegistry
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación FastAPI.

    Al arrancar: carga todos los modelos ML desde W&B, inicializa las cachés,
    descarga metadatos de estaciones desde Google Drive y los datos estáticos
    GTFS, y lanza la tarea de fondo para emitir alertas por WebSocket.

    Al parar: cancela la tarea de fondo de emisión de alertas.

    Parámetros:
        app: Instancia de la aplicación FastAPI.
    """
    logger.info("Starting Express-Bound inference API…")

    registry = ModelRegistry()
    entity = settings.wandb_entity

    # Carga en paralelo: cada loader es network-bound (descarga artefacto W&B)
    # y registra el error internamente, así que asyncio.gather sin return_exceptions
    # es seguro. Cuello de botella original: 7 descargas serializadas en startup.
    loaders = [
        (registry.load_dcrnn,          (entity, settings.wandb_project_dcrnn,   settings.dcrnn_artifact)),
        (registry.load_lgbm_delay_30m, (entity, settings.wandb_project_delay,   settings.lgbm_delay_30m_artifact)),
        (registry.load_lgbm_delay_end, (entity, settings.wandb_project_delay,   settings.lgbm_delay_end_artifact)),
        (registry.load_delta_10m,      (entity, settings.wandb_project_delay,   settings.delta_10m_artifact)),
        (registry.load_delta_20m,      (entity, settings.wandb_project_delay,   settings.delta_20m_artifact)),
        (registry.load_delta_30m,      (entity, settings.wandb_project_delay,   settings.delta_30m_artifact)),
        (registry.load_alertas,        (entity, settings.wandb_project_alertas, settings.alertas_artifact)),
    ]
    await asyncio.gather(*(asyncio.to_thread(loader, *args) for loader, args in loaders))

    if registry.errors:
        logger.warning("Some models failed to load: %s", list(registry.errors.keys()))
    else:
        logger.info("All models loaded successfully")

    app.state.registry = registry
    app.state.cache = TTLCache(ttl_seconds=settings.data_cache_ttl)
    app.state.vehicles_cache = TTLCache(ttl_seconds=90)
    app.state.stations_meta = _load_stations_meta()
    gtfs_static = await asyncio.to_thread(_load_gtfs_static)
    app.state.gtfs_shapes = gtfs_static["shapes"]
    app.state.gtfs_stops = gtfs_static["stops"]
    app.state.prev_stop_for_route = gtfs_static["prev_stop"]
    app.state.ws_manager = _ConnectionManager()
    app.state.bg_task = asyncio.create_task(_live_broadcast(app))

    yield

    app.state.bg_task.cancel()
    logger.info("Express-Bound API stopped")


def _load_stations_meta() -> dict:
    """
    Descarga y parsea los metadatos de las estaciones del metro de NY desde Google Drive.

    Obtiene el fichero MTA_Subway_Stations.csv de la carpeta GTFS_estatic en Drive
    y construye un diccionario indexado por GTFS Stop ID con nombre, coordenadas y
    líneas que pasan por cada estación.

    Retorna:
        Diccionario con clave = stop_id (str) y valor = dict con campos
        'name', 'lat', 'lon' y 'routes'. Retorna {} si el fichero no está disponible.
    """
    from app.data.drive import download_daily_file

    df = None

    try:
        from app.data.drive import _get_service, _get_folder_id, _DEFAULT_TOKEN_PATH
        import io
        from googleapiclient.http import MediaIoBaseDownload
        service = _get_service(_DEFAULT_TOKEN_PATH)
        root_id = _get_folder_id(service, "MTA_Daily_Data")
        folder_id = _get_folder_id(service, "GTFS_estatic", parent_id=root_id)
        result = service.files().list(
            q=f"'{folder_id}' in parents and name = 'MTA_Subway_Stations.csv' and trashed = false",
            fields="files(id, name)", pageSize=1,
        ).execute()
        files = result.get("files", [])
        if not files:
            raise FileNotFoundError("MTA_Subway_Stations.csv no encontrado en Drive/GTFS_estatic")
        file_id = files[0]["id"]
        req = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        df = pd.read_csv(buf)
        logger.info("Station metadata loaded from Drive (%d rows)", len(df))
    except Exception as exc:
        logger.warning("Drive stations not available: %s", exc)

    if df is None:
        logger.warning("Station metadata unavailable — coordinates will be omitted")
        return {}

    meta = {}
    try:
        for _, row in df.iterrows():
            meta[str(row["GTFS Stop ID"])] = {
                "name": row.get("Stop Name", ""),
                "lat": float(row["GTFS Latitude"]),
                "lon": float(row["GTFS Longitude"]),
                "routes": row.get("Daytime Routes", ""),
            }
    except Exception as exc:
        logger.warning("Error parsing station CSV: %s", exc)
    return meta


def _load_gtfs_static() -> dict:
    """
    Descarga y parsea el feed GTFS estático público de la MTA.

    Obtiene el ZIP de google_transit desde la web de la MTA, extrae los ficheros
    trips.txt, shapes.txt, stops.txt y stop_times.txt, y construye tres estructuras:
    - gtfs_shapes: geometría (lista de segmentos lat/lon) por route_id.
    - gtfs_stops: coordenadas (lat, lon) por stop_id.
    - prev_stop_for_route: diccionario (route_id, stop_id) -> stop_id previo, para
      interpolar la posición visual de los trenes en movimiento.

    Si la librería rdp está disponible, simplifica las geometrías con el algoritmo
    Ramer-Douglas-Peucker (epsilon=0.0002 grados, ~22 metros).

    Retorna:
        Diccionario con claves 'shapes', 'stops' y 'prev_stop'. Si la descarga
        falla, retorna estructuras vacías (no interrumpe el arranque del servidor).
    """
    url = "http://web.mta.info/developers/data/nyct/subway/google_transit.zip"

    VALID_ROUTES = {
        '1','2','3','4','5','6','7',
        'A','C','E','B','D','F','M',
        'G','J','Z','L','N','Q','R','W',
        'S','GS','FS','H','SIR'
    }

    def normalize_route_id(rid: str) -> str | None:
        """
        Normaliza un route_id del GTFS al identificador canónico de la línea.

        Parámetros:
            rid: Identificador de ruta crudo del GTFS (puede contener sufijos o guiones).

        Retorna:
            El route_id normalizado si es válido, None en caso contrario.
        """
        rid = rid.strip()
        if rid in VALID_ROUTES:
            return rid
        base = rid.split('-')[0].split('_')[0]
        if base in VALID_ROUTES:
            return base
        if base == 'SI':
            return 'SIR'
        return None

    empty = {"shapes": {}, "stops": {}, "prev_stop": {}}

    try:
        logger.info("Downloading MTA GTFS static feed from %s …", url)
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            content = resp.read()
        logger.info("GTFS zip downloaded (%d bytes), parsing…", len(content))

        zf = zipfile.ZipFile(io.BytesIO(content))

        trips = pd.read_csv(zf.open("trips.txt"))
        shapes_df = pd.read_csv(zf.open("shapes.txt"))
        shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

        trips["route_id_norm"] = trips["route_id"].astype(str).map(normalize_route_id)
        trips = trips[trips["route_id_norm"].notna()]

        shapes_by_id = {
            sid: sub[["shape_pt_lat", "shape_pt_lon"]].values.tolist()
            for sid, sub in shapes_df.groupby("shape_id")
        }

        def geo_extent(shape_id: str) -> float:
            """Extensión geográfica de un shape: (max_lat-min_lat) + (max_lon-min_lon)."""
            pts = shapes_by_id.get(shape_id, [])
            if not pts:
                return 0.0
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            return (max(lats) - min(lats)) + (max(lons) - min(lons))

        def endpoint_key(shape_id: str) -> tuple | None:
            """Clave (inicio, fin) redondeada a ~1 km para agrupar shapes del mismo ramal."""
            pts = shapes_by_id.get(shape_id, [])
            if len(pts) < 2:
                return None
            s = (round(pts[0][0], 2), round(pts[0][1], 2))
            e = (round(pts[-1][0], 2), round(pts[-1][1], 2))
            return (min(s, e), max(s, e))

        gtfs_shapes: dict[str, list] = {}
        for route_id, grp in trips.groupby("route_id_norm"):
            # Un shape representativo por ramal único (agrupado por endpoints).
            # Ordenar por extensión geográfica garantiza que elegimos el shape más
            # completo de cada ramal, en lugar del primero arbitrario, lo que evita
            # excluir ramales cortos como A→Lefferts Blvd.
            unique_shapes = sorted(grp["shape_id"].unique(), key=geo_extent, reverse=True)
            seen: dict[tuple, str] = {}
            for shape_id in unique_shapes:
                key = endpoint_key(shape_id)
                if key and key not in seen:
                    seen[key] = shape_id
            segments = []
            for shape_id in seen.values():
                pts = shapes_by_id.get(shape_id, [])
                if len(pts) >= 2:
                    segments.append(pts)
            if segments:
                gtfs_shapes[str(route_id)] = segments

        try:
            from rdp import rdp as rdp_simplify
            simplified_shapes = {}
            for route_id, segments in gtfs_shapes.items():
                simplified_segs = []
                for seg in segments:
                    if len(seg) > 10:
                        # Simplificación RDP con epsilon=0.0002 grados (~22 metros)
                        simplified_seg = rdp_simplify(seg, 0.0002)
                        simplified_segs.append(simplified_seg)
                    else:
                        simplified_segs.append(seg)
                simplified_shapes[route_id] = simplified_segs

            orig_pts = sum(len(seg) for segs in gtfs_shapes.values() for seg in segs)
            simp_pts = sum(len(seg) for segs in simplified_shapes.values() for seg in segs)
            logger.info(f"GTFS shapes simplified: {orig_pts} → {simp_pts} points ({100*simp_pts/orig_pts:.1f}%)")
            gtfs_shapes = simplified_shapes
        except Exception as e:
            logger.warning(f"Could not simplify shapes: {e}, using unsimplified")

        logger.info("GTFS shapes loaded for %d routes: %s", len(gtfs_shapes), sorted(gtfs_shapes.keys()))

        stops_df = pd.read_csv(
            zf.open("stops.txt"),
            usecols=["stop_id", "stop_lat", "stop_lon"],
            dtype={"stop_id": str},
        )
        gtfs_stops: dict[str, tuple[float, float]] = {
            row.stop_id: (float(row.stop_lat), float(row.stop_lon))
            for row in stops_df.itertuples()
        }
        logger.info("GTFS stops loaded: %d stops", len(gtfs_stops))

        st_df = pd.read_csv(
            zf.open("stop_times.txt"),
            usecols=["trip_id", "stop_sequence", "stop_id"],
            dtype={"trip_id": str, "stop_id": str, "stop_sequence": int},
        )
        trip_route_dir = trips[["trip_id", "route_id_norm", "direction_id"]].drop_duplicates()
        st_merged = st_df.merge(trip_route_dir, on="trip_id")

        prev_stop_for_route: dict[tuple[str, str], str] = {}
        for (route_id_norm, direction_id), grp in st_merged.groupby(["route_id_norm", "direction_id"]):
            best_trip = grp.groupby("trip_id")["stop_sequence"].count().idxmax()
            ordered = (grp[grp["trip_id"] == best_trip]
                       .sort_values("stop_sequence")["stop_id"]
                       .tolist())
            for i in range(1, len(ordered)):
                key = (route_id_norm, ordered[i])
                if key not in prev_stop_for_route:
                    prev_stop_for_route[key] = ordered[i - 1]

        logger.info("prev_stop_for_route built: %d entries", len(prev_stop_for_route))

        return {"shapes": gtfs_shapes, "stops": gtfs_stops, "prev_stop": prev_stop_for_route}

    except Exception as exc:
        logger.warning("Could not load GTFS static data (non-fatal): %s", exc)
        return empty


def _sanitize_json(obj):
    """
    Elimina valores float no serializables (NaN, Inf) reemplazándolos por None.

    Recorre recursivamente dicts y listas. Necesario antes de serializar respuestas
    a JSON cuando los modelos ML pueden producir valores infinitos o NaN.

    Parámetros:
        obj: Cualquier objeto Python (float, dict, list u otro tipo).

    Retorna:
        El mismo objeto con los floats no finitos sustituidos por None.
    """
    import math
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


async def _live_broadcast(app: FastAPI) -> None:
    """
    Tarea de fondo que emite alertas a todos los clientes WebSocket cada 60 segundos.

    Utiliza las ventanas de datos cacheadas y el modelo de alertas para generar
    predicciones y las envía como JSON a través del gestor de conexiones WebSocket.
    Los errores no fatales se registran como DEBUG y el bucle continúa.

    Parámetros:
        app: Instancia de la aplicación FastAPI (para acceder a app.state).
    """
    while True:
        await asyncio.sleep(60)
        try:
            # Si no hay clientes conectados, no merece la pena correr el modelo:
            # el resultado se descartaría inmediatamente. Ahorra una inferencia
            # por minuto en idle.
            if not app.state.ws_manager.has_clients():
                continue

            from app.models.alertas_infer import run_alerts

            registry = app.state.registry
            windows = app.state.cache.get("windows")
            if windows is None or registry.alertas is None:
                continue

            alerts = await asyncio.to_thread(
                run_alerts,
                entry=registry.alertas,
                windows=windows,
                threshold=settings.alert_threshold,
            )
            payload = json.dumps(_sanitize_json({
                "type": "update",
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "alerts": alerts.model_dump(),
            }))
            await app.state.ws_manager.broadcast(payload)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Live broadcast error (non-fatal): %s", exc)


class _ConnectionManager:
    """
    Gestiona las conexiones WebSocket activas y el envío de mensajes en broadcast.

    Mantiene una lista interna de conexiones abiertas. Las conexiones que fallen
    durante el broadcast se eliminan automáticamente.
    """

    def __init__(self):
        """Inicializa el gestor con una lista vacía de conexiones."""
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        """
        Acepta una nueva conexión WebSocket y la añade a la lista activa.

        Parámetros:
            ws: Objeto WebSocket de FastAPI a registrar.
        """
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        """
        Elimina una conexión WebSocket de la lista activa.

        Parámetros:
            ws: Objeto WebSocket a desregistrar.
        """
        if ws in self._connections:
            self._connections.remove(ws)

    def has_clients(self) -> bool:
        """Devuelve True si hay al menos una conexión WebSocket activa."""
        return bool(self._connections)

    async def broadcast(self, message: str) -> None:
        """
        Envía un mensaje de texto a todas las conexiones WebSocket activas.

        Las conexiones que fallen durante el envío se marcan como muertas y se
        eliminan al finalizar el broadcast.

        Parámetros:
            message: Texto (normalmente JSON serializado) a enviar.
        """
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


app = FastAPI(
    title="Express-Bound",
    description="Real-time MTA subway delay prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# Comprime respuestas JSON > 1 KiB. /api/shapes y /api/routes son de varios cientos
# de KiB y se reducen ~3-4× con gzip; los WebSockets quedan exentos por defecto.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(health_router)
app.include_router(predict_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """
    Sirve la página principal de la interfaz web.

    Parámetros:
        request: Objeto Request de FastAPI.

    Retorna:
        Respuesta HTML renderizada con la plantilla index.html.
    """
    return templates.TemplateResponse(request, name="index.html")


_STATIC_CACHE_HEADER = "public, max-age=3600"


@app.get("/api/stations")
def get_stations(request: Request, response: Response):
    """
    Devuelve la lista completa de estaciones con sus metadatos.

    Parámetros:
        request: Objeto Request de FastAPI (accede a app.state.stations_meta).
        response: Para fijar Cache-Control (datos construidos una vez en startup).

    Retorna:
        Lista de dicts con 'id', 'name', 'lat', 'lon' y 'routes' por estación.
    """
    response.headers["Cache-Control"] = _STATIC_CACHE_HEADER
    return [
        {"id": sid, **data}
        for sid, data in request.app.state.stations_meta.items()
    ]


@app.get("/api/warmup")
async def warmup(request: Request):
    """
    Pre-carga la caché de ventanas de Drive para evitar latencia extra en la primera predicción.

    Si las ventanas no están cacheadas, las descarga de Google Drive y las almacena
    en la caché TTL de la aplicación.

    Parámetros:
        request: Objeto Request de FastAPI.

    Retorna:
        Dict con {"status": "ready"} una vez que la caché está disponible.
    """
    from app.data.drive import download_windows
    cache = request.app.state.cache
    if cache.get("windows") is None:
        windows = await asyncio.to_thread(
            download_windows,
            n_windows=settings.n_windows,
            token_path=settings.drive_token_path,
            folder_name=settings.google_drive_folder_name,
        )
        cache.set("windows", windows)
    return {"status": "ready"}


@app.get("/api/debug/stop")
async def debug_stop(request: Request, stop_id: str):
    """
    Endpoint de diagnóstico para verificar la cobertura de un stop_id concreto.

    Comprueba si el stop_id aparece en los nodos del modelo DCRNN y en la última
    ventana de datos cacheada de Drive, mostrando información detallada de cobertura.

    Parámetros:
        request: Objeto Request de FastAPI.
        stop_id: Identificador GTFS de la parada a diagnosticar.

    Retorna:
        Dict con secciones 'dcrnn' y 'drive_windows' describiendo la cobertura encontrada.
    """
    registry = request.app.state.registry
    cache = request.app.state.cache
    windows = cache.get("windows")

    result: dict = {"stop_id": stop_id, "dcrnn": {}, "drive_windows": {}}

    if registry.dcrnn is not None:
        nodes = registry.dcrnn.nodes
        def _gtfs_stop(n: str) -> str:
            parts = n.split("_", 1)
            return parts[1] if len(parts) > 1 else n
        exact = [n for n in nodes if _gtfs_stop(n) == stop_id]
        base_match = [n for n in nodes if _gtfs_stop(n)[:-1] == stop_id
                      and _gtfs_stop(n)[-1] in ("N", "S")]
        result["dcrnn"] = {
            "n_total_nodes": len(nodes),
            "exact_match": exact,
            "directional_match": base_match,
            "covered": bool(exact or base_match),
        }
    else:
        result["dcrnn"] = {"error": "DCRNN not loaded"}

    if windows:
        df = windows[-1]
        stop_ids_in_window = df["stop_id"].astype(str).unique().tolist() if "stop_id" in df.columns else []
        base_ids = [s.rstrip("NS") for s in stop_ids_in_window]
        exact_rows = df[df["stop_id"].astype(str) == stop_id] if "stop_id" in df.columns else df.iloc[0:0]
        base_rows = df[df["stop_id"].astype(str).str.rstrip("NS") == stop_id] if "stop_id" in df.columns else df.iloc[0:0]
        found = exact_rows if not exact_rows.empty else base_rows
        result["drive_windows"] = {
            "n_windows": len(windows),
            "n_rows_latest_window": len(df),
            "stop_found": not found.empty,
            "matching_stop_ids": found["stop_id"].astype(str).unique().tolist() if not found.empty else [],
            "routes_in_window": df["route_id"].astype(str).unique().tolist() if "route_id" in df.columns else [],
            "delay_seconds_mean_sample": found["delay_seconds_mean"].tolist() if (not found.empty and "delay_seconds_mean" in found.columns) else [],
        }
    else:
        result["drive_windows"] = {"error": "No windows cached — call /api/warmup first"}

    return result


# Orden canónico de paradas por línea, de norte a sur o de inicio a fin de recorrido.
_ROUTE_ORDER: dict[str, list[str]] = {
    "1": ["Van Cortlandt Park-242 St","238 St","231 St","Marble Hill-225 St","215 St","207 St","Dyckman St","191 St","181 St","168 St-Washington Hts","157 St","145 St","137 St-City College","125 St","116 St-Columbia University","Cathedral Pkwy (110 St)","103 St","96 St","86 St","79 St","72 St","66 St-Lincoln Center","59 St-Columbus Circle","50 St","Times Sq-42 St","34 St-Penn Station","28 St","23 St","18 St","14 St","Christopher St-Stonewall","Houston St","Canal St","Franklin St","Chambers St","WTC Cortlandt","Rector St","South Ferry"],
    "2": ["Wakefield-241 St","Nereid Av","233 St","225 St","219 St","Gun Hill Rd","Burke Av","Allerton Av","Pelham Pkwy","Bronx Park East","E 180 St","West Farms Sq-E Tremont Av","174 St","Freeman St","Simpson St","Intervale Av","Prospect Av","Jackson Av","3 Av-149 St","149 St-Grand Concourse","135 St","125 St","116 St","110 St-Malcolm X Plaza","103 St","96 St","86 St","72 St","Times Sq-42 St","34 St-Penn Station","28 St","23 St","14 St","Chambers St","Fulton St","Wall St","Clark St","Borough Hall","Nevins St","Atlantic Av-Barclays Ctr","Bergen St","Carroll St","Smith-9 Sts","4 Av-9 St","Prospect Av","25 St","36 St","53 St","59 St"],
    "3": ["Harlem-148 St","145 St","135 St","125 St","116 St","110 St-Malcolm X Plaza","96 St","72 St","Times Sq-42 St","34 St-Penn Station","14 St","Chambers St","Park Place","Fulton St","Wall St","Clark St","Borough Hall","Nevins St","Atlantic Av-Barclays Ctr","Grand Army Plaza","Eastern Pkwy-Brooklyn Museum","Crown Hts-Utica Av","Sutter Av-Rutland Rd","Saratoga Av","Rockaway Av","New Lots Av"],
    "4": ["Woodlawn","Mosholu Pkwy","Bedford Park Blvd","Kingsbridge Rd","Fordham Rd","183 St","Burnside Av","176 St","Mt Eden Av","170 St","167 St","161 St-Yankee Stadium","149 St-Grand Concourse","138 St-Grand Concourse","125 St","116 St","110 St","103 St","86 St","77 St","68 St-Hunter College","59 St","51 St","Grand Central-42 St","33 St","28 St","23 St-Baruch College","14 St-Union Sq","Brooklyn Bridge-City Hall","Fulton St","Wall St","Bowling Green","Borough Hall","Nevins St","Atlantic Av-Barclays Ctr","Franklin Av-Medgar Evers College","Crown Hts-Utica Av","Sutter Av-Rutland Rd","Saratoga Av","Junius St","Pennsylvania Av","Van Siclen Av","New Lots Av"],
    "5": ["Eastchester-Dyre Av","Baychester Av","Gun Hill Rd","Pelham Pkwy","Morris Park","E 180 St","West Farms Sq-E Tremont Av","174 St","Freeman St","Simpson St","Intervale Av","Prospect Av","Jackson Av","3 Av-149 St","149 St-Grand Concourse","125 St","116 St","110 St","103 St","86 St","77 St","68 St-Hunter College","59 St","51 St","Grand Central-42 St","33 St","28 St","23 St-Baruch College","14 St-Union Sq","Brooklyn Bridge-City Hall","Fulton St","Wall St","Bowling Green","Borough Hall","Nevins St","Atlantic Av-Barclays Ctr","Franklin Av-Medgar Evers College","Crown Hts-Utica Av","Sutter Av-Rutland Rd","Saratoga Av","Junius St","Pennsylvania Av","Van Siclen Av","New Lots Av"],
    "6": ["Pelham Bay Park","Buhre Av","Middletown Rd","Westchester Sq-E Tremont Av","Zerega Av","Castle Hill Av","Parkchester","St Lawrence Av","Morrison Av-Soundview","Elder Av","Whitlock Av","Hunts Point Av","Longwood Av","E 149 St","E 143 St-St Mary's St","Cypress Av","3 Av-138 St","125 St","116 St","110 St","103 St","96 St","86 St","77 St","68 St-Hunter College","59 St","51 St","Grand Central-42 St","33 St","28 St","23 St-Baruch College","14 St-Union Sq","Astor Pl","Bleecker St","Spring St","Canal St","Brooklyn Bridge-City Hall"],
    "7": ["Flushing-Main St","Mets-Willets Point","111 St","103 St-Corona Plaza","Junction Blvd","90 St-Elmhurst Av","82 St-Jackson Hts","74 St-Broadway","69 St","61 St-Woodside","52 St","46 St-Bliss St","40 St-Lowery St","33 St-Rawson St","Queensboro Plaza","Court Sq","Hunters Point Av","Vernon Blvd-Jackson Av","Grand Central-42 St","5 Av","Times Sq-42 St","34 St-Hudson Yards"],
    "A": ["Inwood-207 St","Dyckman St","190 St","181 St","175 St","168 St","163 St-Amsterdam Av","155 St","145 St","125 St","59 St-Columbus Circle","50 St","42 St-Port Authority Bus Terminal","34 St-Penn Station","23 St","14 St","W 4 St-Wash Sq","Spring St","Canal St","Chambers St","Fulton St","High St","Jay St-MetroTech","Hoyt-Schermerhorn Sts","Lafayette Av","Clinton-Washington Avs","Franklin Av","Nostrand Av","Kingston-Throop Avs","Ralph Av","Utica Av","Rockaway Av","Broadway Junction","Atlantic Av","Ozone Park-Lefferts Blvd"],
    "C": ["168 St","163 St-Amsterdam Av","155 St","145 St","135 St","125 St","116 St","Cathedral Pkwy (110 St)","103 St","96 St","86 St","81 St-Museum of Natural History","72 St","59 St-Columbus Circle","50 St","42 St-Port Authority Bus Terminal","34 St-Penn Station","23 St","14 St","W 4 St-Wash Sq","Spring St","Canal St","Chambers St","Fulton St","High St","Jay St-MetroTech","Hoyt-Schermerhorn Sts","Lafayette Av","Clinton-Washington Avs","Franklin Av","Nostrand Av","Kingston-Throop Avs","Ralph Av","Utica Av","Rockaway Av","Shepherd Av","Euclid Av"],
    "E": ["Jamaica Center-Parsons/Archer","Sutphin Blvd-Archer Av-JFK Airport","Jamaica-Van Wyck","Kew Gardens-Union Tpke","Briarwood","Forest Hills-71 Av","75 Av","Jackson Hts-Roosevelt Av","65 St","Woodhaven Blvd","85 St-Forest Pkwy","104 St","111 St","Queens Plaza","Court Sq-23 St","Lexington Av/53 St","5 Av/53 St","7 Av","50 St","42 St-Port Authority Bus Terminal","34 St-Penn Station","23 St","14 St","W 4 St-Wash Sq","Spring St","Canal St","Chambers St","World Trade Center"],
    "B": ["145 St","135 St","125 St","116 St","Cathedral Pkwy (110 St)","103 St","96 St","86 St","81 St-Museum of Natural History","72 St","59 St-Columbus Circle","7 Av","47-50 Sts-Rockefeller Ctr","42 St-Bryant Pk","34 St-Herald Sq","23 St","14 St","W 4 St-Wash Sq","Broadway-Lafayette St","Grand St","DeKalb Av","Atlantic Av-Barclays Ctr","7 Av","15 St-Prospect Park","Church Av","Cortelyou Rd","Newkirk Plaza","Avenue H","Avenue J","Avenue M","Kings Hwy","Avenue U","Neck Rd","Sheepshead Bay","Brighton Beach","Ocean Pkwy","Coney Island-Stillwell Av"],
    "D": ["Norwood-205 St","Mosholu Pkwy","Bedford Park Blvd-Lehman College","Kingsbridge Rd","Fordham Rd","182-183 Sts","Tremont Av","174-175 Sts","170 St","167 St","161 St-Yankee Stadium","155 St","145 St","125 St","116 St","Cathedral Pkwy (110 St)","103 St","96 St","86 St","81 St-Museum of Natural History","72 St","59 St-Columbus Circle","7 Av","47-50 Sts-Rockefeller Ctr","42 St-Bryant Pk","34 St-Herald Sq","23 St","14 St","W 4 St-Wash Sq","Broadway-Lafayette St","Grand St","DeKalb Av","Atlantic Av-Barclays Ctr","36 St","New Utrecht Av","18 Av","20 Av","Bay Pkwy","25 Av","Bay 50 St","Coney Island-Stillwell Av"],
    "F": ["Jamaica-179 St","169 St","Parsons Blvd","Sutphin Blvd","Briarwood","Kew Gardens-Union Tpke","75 Av","Forest Hills-71 Av","67 Av","Jackson Hts-Roosevelt Av","65 St","Woodhaven Blvd","85 St-Forest Pkwy","104 St","111 St","Queens Plaza","Court Sq-23 St","Lexington Av/63 St","21 St-Queensbridge","Roosevelt Island","47-50 Sts-Rockefeller Ctr","42 St-Bryant Pk","34 St-Herald Sq","23 St","14 St","W 4 St-Wash Sq","2 Av","Broadway-Lafayette St","East Broadway","Delancey St-Essex St","Bergen St","Carroll St","Smith-9 Sts","4 Av-9 St","7 Av","15 St-Prospect Park","Church Av","Ditmas Av","18 Av","Avenue I","Bay Pkwy","Avenue N","Avenue P","Kings Hwy","Avenue U","Avenue X","Neptune Av","W 8 St-NY Aquarium","Coney Island-Stillwell Av"],
    "G": ["Court Sq","21 St","Queensboro Plaza","Nassau Av","Greenpoint Av","Metropolitan Av","Broadway","Flushing Av","Myrtle-Willoughby Avs","Bedford-Nostrand Avs","Classon Av","Clinton-Washington Avs","Fulton St","Lafayette Av","Atlantic Av-Barclays Ctr","7 Av","15 St-Prospect Park","Church Av","Fort Hamilton Pkwy","Smith-9 Sts","4 Av-9 St","Carroll St","Bergen St","Hoyt-Schermerhorn Sts","Jay St-MetroTech"],
    "J": ["Jamaica Center-Parsons/Archer","Sutphin Blvd-Archer Av-JFK Airport","Jamaica-Van Wyck","121 St","111 St","104 St","85 St-Forest Pkwy","Woodhaven Blvd","75 St-Elderts Ln","Cypress Hills","Crescent St","Norwood Av","Cleveland St","Van Siclen Av","Alabama Av","Broadway Junction","Halsey St","Gates Av","Myrtle Av","Kosciuszko St","Flushing Av","Hewes St","Marcy Av","Lorimer St","Delancey St-Essex St","Bowery","Canal St","Chambers St","Fulton St","Broad St"],
    "Z": ["Jamaica Center-Parsons/Archer","Sutphin Blvd-Archer Av-JFK Airport","Jamaica-Van Wyck","121 St","111 St","104 St","85 St-Forest Pkwy","Woodhaven Blvd","75 St-Elderts Ln","Cypress Hills","Crescent St","Norwood Av","Cleveland St","Van Siclen Av","Alabama Av","Broadway Junction","Halsey St","Gates Av","Myrtle Av","Kosciuszko St","Flushing Av","Hewes St","Marcy Av","Lorimer St","Delancey St-Essex St","Bowery","Canal St","Chambers St","Fulton St","Broad St"],
    "L": ["8 Av","6 Av","14 St-Union Sq","3 Av","1 Av","Bedford Av","Lorimer St","Graham Av","Grand St","Montrose Av","Morgan Av","Jefferson St","DeKalb Av","Myrtle-Wyckoff Avs","Halsey St","Wilson Av","Bushwick Av-Aberdeen St","Broadway Junction","Atlantic Av","Sutter Av","Livonia Av","New Lots Av","East 105 St","Canarsie-Rockaway Pkwy"],
    "M": ["Middle Village-Metropolitan Av","Fresh Pond Rd","Forest Av","Seneca Av","Forest Av","Woodhaven Blvd","85 St-Forest Pkwy","104 St","111 St","Queens Plaza","Lexington Av/53 St","5 Av/53 St","7 Av","47-50 Sts-Rockefeller Ctr","42 St-Bryant Pk","34 St-Herald Sq","23 St","14 St","W 4 St-Wash Sq","Broadway-Lafayette St","Grand St","DeKalb Av","Atlantic Av-Barclays Ctr"],
    "N": ["Astoria-Ditmars Blvd","Astoria Blvd","30 Av","Broadway","36 Av","39 Av-Dutch Kills","Queensboro Plaza","Queens Plaza","Lexington Av/59 St","5 Av/59 St","57 St-7 Av","49 St","Times Sq-42 St","34 St-Herald Sq","28 St","23 St","14 St","8 St-NYU","Prince St","Canal St","City Hall","WTC Cortlandt","Rector St","Whitehall St-South Ferry","Court St","DeKalb Av","Atlantic Av-Barclays Ctr","36 St","New Utrecht Av","18 Av","20 Av","Bay Pkwy","25 Av","Bay 50 St","Coney Island-Stillwell Av"],
    "Q": ["96 St","86 St","72 St","57 St-7 Av","49 St","Times Sq-42 St","34 St-Herald Sq","28 St","23 St","14 St","8 St-NYU","Prince St","Canal St","DeKalb Av","Atlantic Av-Barclays Ctr","7 Av","15 St-Prospect Park","Church Av","Cortelyou Rd","Newkirk Plaza","Newkirk Av-Little Haiti","Avenue H","Avenue J","Avenue M","Kings Hwy","Avenue U","Neck Rd","Sheepshead Bay","Brighton Beach","Ocean Pkwy","Coney Island-Stillwell Av"],
    "R": ["Forest Hills-71 Av","67 Av","63 Dr-Rego Park","Woodhaven Blvd","85 St-Forest Pkwy","104 St","111 St","Queens Plaza","Lexington Av/59 St","5 Av/59 St","57 St-7 Av","49 St","Times Sq-42 St","34 St-Herald Sq","28 St","23 St","14 St","8 St-NYU","Prince St","Canal St","City Hall","WTC Cortlandt","Rector St","Whitehall St-South Ferry","Court St","DeKalb Av","Atlantic Av-Barclays Ctr","Union St","4 Av-9 St","Prospect Av","25 St","36 St","45 St","53 St","59 St","Bay Ridge Av","77 St","86 St","Bay Ridge-95 St"],
    "W": ["Astoria-Ditmars Blvd","Astoria Blvd","30 Av","Broadway","36 Av","39 Av-Dutch Kills","Queensboro Plaza","Queens Plaza","Lexington Av/59 St","5 Av/59 St","57 St-7 Av","49 St","Times Sq-42 St","34 St-Herald Sq","28 St","23 St","14 St","8 St-NYU","Prince St","Canal St","City Hall","WTC Cortlandt","Rector St","Whitehall St-South Ferry"],
    "S": ["Times Sq-42 St","Grand Central-42 St"],
    "SIR": ["St George","Tompkinsville","Stapleton","Clifton","Grasmere","Old Town","Dongan Hills","Jefferson Av","Grant City","New Dorp","Oakwood Heights","Bay Terrace","Great Kills","Eltingville","Annadale","Huguenot","Prince's Bay","Richmond Valley","Arthur Kill","Tottenville"],
}


@app.get("/api/routes")
def get_routes(request: Request, response: Response):
    """
    Devuelve el orden canónico de stop_ids por línea para el mapa de la interfaz.

    Construye un índice inverso de nombre de parada a stop_id usando los metadatos
    de estaciones, luego recorre _ROUTE_ORDER para asignar los IDs correspondientes.

    Parámetros:
        request: Objeto Request de FastAPI.
        response: Para fijar Cache-Control (datos derivados de stations_meta estático).

    Retorna:
        Dict con clave = route_id y valor = lista ordenada de stop_ids.
        Solo incluye líneas con al menos 2 paradas resueltas.
    """
    response.headers["Cache-Control"] = _STATIC_CACHE_HEADER
    meta = request.app.state.stations_meta

    name_to_candidates: dict[str, list[tuple[str, set[str]]]] = {}
    for sid, data in meta.items():
        name = str(data.get("name", "")).strip().lower()
        routes_set = set(str(data.get("routes", "")).split())
        name_to_candidates.setdefault(name, []).append((sid, routes_set))

    def lookup(stop_name: str, line: str) -> str | None:
        """
        Busca el stop_id que mejor coincide con el nombre y línea dados.

        Parámetros:
            stop_name: Nombre de la parada (sin normalizar).
            line: Identificador de la línea (p.ej. 'A', '1').

        Retorna:
            El stop_id más apropiado, o None si no se encuentra ninguno.
        """
        candidates = name_to_candidates.get(stop_name.lower(), [])
        if not candidates:
            return None
        for sid, routes_set in candidates:
            if line in routes_set:
                return sid
        return candidates[0][0]

    result = {}
    for line, stop_names in _ROUTE_ORDER.items():
        ids = [sid for n in stop_names if (sid := lookup(n, line)) is not None]
        if len(ids) >= 2:
            result[line] = ids
    return result


@app.get("/api/shapes")
def get_shapes(request: Request, response: Response):
    """
    Devuelve la geometría GTFS de cada línea para renderizar el mapa.

    Parámetros:
        request: Objeto Request de FastAPI.
        response: Para fijar Cache-Control (geometría estática cargada en startup).

    Retorna:
        Dict con clave = route_id y valor = lista de segmentos [[lat, lon], ...].
    """
    response.headers["Cache-Control"] = _STATIC_CACHE_HEADER
    return request.app.state.gtfs_shapes


@app.get("/api/vehicles")
async def get_vehicles(request: Request) -> list[dict]:
    """
    Devuelve las posiciones actuales de los trenes en servicio, con caché de 90 segundos.

    Si la caché no ha expirado, devuelve los datos almacenados. En caso contrario,
    consulta los feeds GTFS-RT de la MTA y actualiza la caché.

    Parámetros:
        request: Objeto Request de FastAPI.

    Retorna:
        Lista de dicts con posición, ruta, dirección y estado de cada tren.
    """
    cache = request.app.state.vehicles_cache
    cached = cache.get("vehicles")
    if cached is not None:
        return cached

    positions = await asyncio.to_thread(
        fetch_positions,
        request.app.state.gtfs_stops,
        request.app.state.prev_stop_for_route,
    )
    cache.set("vehicles", positions)
    return positions


@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket para recibir alertas en tiempo real.

    Registra la conexión en el gestor y mantiene el bucle de recepción activo
    hasta que el cliente se desconecte.

    Parámetros:
        websocket: Objeto WebSocket de FastAPI.
    """
    manager: _ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
