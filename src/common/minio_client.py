"""
Cliente simplificado para subir y descargar datos desde MinIO.

Proporciona funciones de alto nivel para las operaciones más habituales
del proyecto: transferencia de archivos locales, DataFrames serializados
como Parquet y objetos JSON.

Dependencias:
    - minio
    - pandas
    - pyarrow (necesario para la serialización Parquet)

Valores por defecto:
    endpoint : "minio.fdi.ucm.es"
    bucket   : "pd1"

Uso:
    uv run python src/common/minio_client.py

Casos de uso:

1) Subir/Descargar archivo local:
   upload_file(access_key, secret_key, object_name, file_path,
                    endpoint, bucket)
   download_file(access_key, secret_key, object_name, file_path,
                    endpoint, bucket)

2) Subir DataFrame como Parquet:
   upload_df_parquet(access_key, secret_key, object_name, df,
                        endpoint, bucket)

3) Descargar Parquet como DataFrame:
   df = download_df_parquet(access_key, secret_key, object_name,
                                endpoint, bucket)

4) Subir / descargar JSON:
   upload_json(access_key, secret_key, object_name, data,
                    endpoint, bucket)
   data = download_json(access_key, secret_key, object_name,
                        endpoint, bucket)
"""

import io
import json
from typing import Any

import pandas as pd
from minio import Minio

DEFAULT_ENDPOINT = "minio.fdi.ucm.es"
DEFAULT_BUCKET = "pd1"

def _client(access_key: str, secret_key: str, endpoint: str = DEFAULT_ENDPOINT) -> Minio:
    """
    Crea y devuelve un cliente autenticado de MinIO.

    Parámetros:
        access_key : Clave de acceso de MinIO.
        secret_key : Clave secreta de MinIO.
        endpoint   : Dirección del servidor MinIO (por defecto minio.fdi.ucm.es).

    Retorna:
        Instancia de Minio lista para usar.
    """
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)


def upload_file(
    access_key: str,
    secret_key: str,
    object_name: str,
    file_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """
    Sube un archivo local a MinIO.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta de destino del objeto en MinIO.
        file_path   : Ruta local del archivo a subir.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de destino.
    """
    c = _client(access_key, secret_key, endpoint)
    c.fput_object(bucket, object_name, file_path)


def download_file(
    access_key: str,
    secret_key: str,
    object_name: str,
    file_path: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """
    Descarga un objeto de MinIO a una ruta local.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta del objeto en MinIO.
        file_path   : Ruta local donde se guardará el archivo descargado.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de origen.
    """
    c = _client(access_key, secret_key, endpoint)
    c.fget_object(bucket, object_name, file_path)


def upload_df_parquet(
    access_key: str,
    secret_key: str,
    object_name: str,
    df: pd.DataFrame,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """
    Serializa un DataFrame de pandas como Parquet y lo sube a MinIO en memoria.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta de destino del objeto en MinIO.
        df          : DataFrame a subir.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de destino.
    """
    c = _client(access_key, secret_key, endpoint)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    c.put_object(bucket, object_name, buf, length=buf.getbuffer().nbytes)


def download_df_parquet(
    access_key: str,
    secret_key: str,
    object_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> pd.DataFrame:
    """
    Descarga un archivo Parquet de MinIO y lo devuelve como DataFrame de pandas.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta del objeto Parquet en MinIO.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de origen.

    Retorna:
        DataFrame con los datos del archivo Parquet descargado.
    """
    c = _client(access_key, secret_key, endpoint)
    resp = c.get_object(bucket, object_name)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return pd.read_parquet(io.BytesIO(data))


def upload_json(
    access_key: str,
    secret_key: str,
    object_name: str,
    data: Any,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """
    Serializa un objeto Python a JSON y lo sube a MinIO.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta de destino del objeto en MinIO.
        data        : Objeto Python serializable como JSON.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de destino.
    """
    c = _client(access_key, secret_key, endpoint)
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    buf = io.BytesIO(raw)
    c.put_object(bucket, object_name, buf, length=len(raw))


def download_json(
    access_key: str,
    secret_key: str,
    object_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> Any:
    """
    Descarga un objeto JSON de MinIO y lo devuelve como objeto Python.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta del objeto JSON en MinIO.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket de origen.

    Retorna:
        Objeto Python resultante de deserializar el JSON descargado.
    """
    c = _client(access_key, secret_key, endpoint)
    resp = c.get_object(bucket, object_name)
    try:
        raw = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return json.loads(raw.decode("utf-8"))


def list_objects(
    access_key: str,
    secret_key: str,
    prefix: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> list[str]:
    """
    Lista los nombres de todos los objetos en MinIO que comienzan por un prefijo dado.

    Parámetros:
        access_key : Clave de acceso de MinIO.
        secret_key : Clave secreta de MinIO.
        prefix     : Prefijo de ruta a buscar (actúa como filtro de "carpeta").
        endpoint   : Servidor MinIO.
        bucket     : Nombre del bucket donde buscar.

    Retorna:
        Lista de cadenas con los object_name de los objetos encontrados.
    """
    c = _client(access_key, secret_key, endpoint)
    return [obj.object_name for obj in c.list_objects(bucket, prefix=prefix, recursive=True)]


def delete_object(
    access_key: str,
    secret_key: str,
    object_name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET
) -> None:
    """
    Elimina un único objeto de MinIO.

    Parámetros:
        access_key  : Clave de acceso de MinIO.
        secret_key  : Clave secreta de MinIO.
        object_name : Ruta del objeto a eliminar en MinIO.
        endpoint    : Servidor MinIO.
        bucket      : Nombre del bucket donde reside el objeto.
    """
    c = _client(access_key, secret_key, endpoint)
    c.remove_object(bucket, object_name)
