"""
Script de utilidad para eliminar en bloque todos los objetos de MinIO
que se encuentren bajo un prefijo (carpeta) determinado.

Funcionamiento:
    1. Conecta con el servidor MinIO usando las credenciales de entorno.
    2. Lista recursivamente todos los objetos bajo CARPETA_A_BORRAR.
    3. Ejecuta un borrado masivo con remove_objects.

Dependencias:
    - minio (cliente oficial de MinIO para Python)

Variables de entorno requeridas:
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY

Nota importante:
    Asegúrate de que CARPETA_A_BORRAR termine en "/" para evitar
    eliminar objetos cuyo nombre comience de forma similar pero no
    pertenezcan al prefijo deseado.
"""
import os
from minio import Minio
from minio.deleteobjects import DeleteObject

ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

client = Minio(
    endpoint="minio.fdi.ucm.es",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
)

BUCKET_NAME = "pd1"

# Ruta exacta de la "carpeta" que se desea vaciar (debe terminar en "/").
CARPETA_A_BORRAR = "grupo5/cleaned/eventos_nyc/dia="

if __name__ == "__main__":
    print(f"Buscando archivos en '{CARPETA_A_BORRAR}' dentro del bucket '{BUCKET_NAME}'...")

    # Listar todos los objetos bajo ese prefijo (recursive=True incluye subcarpetas)
    objetos = client.list_objects(BUCKET_NAME, prefix=CARPETA_A_BORRAR, recursive=True, include_user_meta=True)

    elementos_a_borrar = [DeleteObject(obj.object_name) for obj in objetos]

    if elementos_a_borrar:
        print(f"Se han encontrado {len(elementos_a_borrar)} archivos. Procediendo a borrarlos...")

        # remove_objects devuelve un generador con los errores producidos
        errores = client.remove_objects(BUCKET_NAME, elementos_a_borrar)

        hubo_errores = False
        for error in errores:
            print(f"Error al borrar: {error}")
            hubo_errores = True

        if not hubo_errores:
            print("Carpeta borrada con éxito.")
    else:
        print("La carpeta está vacía o no existe. No se ha borrado nada.")
