"""
Implementación de una caché en memoria con tiempo de vida (TTL).

Proporciona la clase TTLCache, una caché de clave-valor simple que expira
automáticamente las entradas transcurrido el número de segundos configurado.
Se usa en la aplicación para cachear las ventanas de datos de Drive y las
posiciones de los vehículos, reduciendo llamadas externas redundantes.

Dependencias:
- Módulo estándar time (time.monotonic para medir tiempos de forma segura).

Notas:
- No es thread-safe por diseño (la aplicación usa asyncio y ejecuta operaciones
  bloqueantes en hilos separados vía asyncio.to_thread).
- Las entradas expiradas se eliminan en el momento de la lectura (lazy eviction).
"""

import time
from typing import Any, Optional


class TTLCache:
    """
    Caché en memoria con expiración automática por tiempo de vida (TTL).

    Almacena pares clave-valor junto con el timestamp de escritura. Al leer
    una clave, comprueba si ha superado el TTL configurado y, si es así,
    la elimina y devuelve None.
    """

    def __init__(self, ttl_seconds: int = 60):
        """
        Inicializa la caché con el tiempo de vida especificado.

        Parámetros:
            ttl_seconds: Segundos que una entrada permanece válida tras su escritura.
                         Por defecto 60 segundos.
        """
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene el valor asociado a la clave si no ha expirado.

        Parámetros:
            key: Clave de la entrada a recuperar.

        Retorna:
            El valor almacenado si existe y no ha expirado; None en caso contrario.
        """
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Almacena un valor en la caché bajo la clave indicada.

        Registra el timestamp actual (monotónico) para controlar la expiración.

        Parámetros:
            key: Clave bajo la que se almacena el valor.
            value: Valor a guardar (puede ser cualquier objeto Python).
        """
        self._store[key] = (value, time.monotonic())

    def invalidate(self, key: str) -> None:
        """
        Elimina una entrada de la caché de forma inmediata, independientemente del TTL.

        Parámetros:
            key: Clave de la entrada a eliminar. No lanza error si no existe.
        """
        self._store.pop(key, None)

    def timestamp(self, key: str) -> Optional[float]:
        """
        Devuelve el timestamp (time.monotonic) en que se almacenó la clave.

        Útil para calcular cuándo fue cacheado un valor en tiempo de reloj de pared.

        Parámetros:
            key: Clave de la entrada.

        Retorna:
            El valor de time.monotonic() en el momento de la escritura, o None
            si la clave no existe (puede haber expirado o nunca haber sido escrita).
        """
        if key not in self._store:
            return None
        _, ts = self._store[key]
        return ts
