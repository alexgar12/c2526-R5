'''
Ejemplo de uso del API de MinIO para subir y descargar archivos.
'''

from minio import Minio
import os

# Leer credenciales de MinIO desde variables de entorno
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
assert ACCESS_KEY is not None, "La variable de entorno MINIO_ACCESS_KEY no está definida."
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
assert SECRET_KEY is not None, "La variable de entorno MINIO_SECRET_KEY no está definida."

# Crear un cliente de MinIO
client = Minio(
    endpoint="minio.fdi.ucm.es",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
)

# Subir el archivo f3.txt a pd1/comun/f3.txt
bucket = "pd1"
source_file = "pruebaSubir2.txt"
destination_file = "comun/pruebaSubir2.txt"
client.fput_object(
  bucket_name=bucket,
  object_name=destination_file,
  file_path=source_file,
)

print(f"Archivo '{source_file}' subido a MinIO como '{destination_file}' en el bucket '{bucket}'.")

# Desgarcar el archivo pd1/comun/f3.txt de MinIO como f4.txt
source_file = "comun/pruebaSubir2.txt"
destination_file = "traidoDeVuelta2.txt"
client.fget_object(
  bucket_name=bucket,
  object_name=source_file,
  file_path=destination_file,
)

print(f"Archivo '{source_file}' bajado del bucket '{bucket}' de MinIO como '{destination_file}'.")

