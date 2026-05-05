"""
Extracción de alertas oficiales de la MTA en tiempo real mediante la API de Gmail.

Lee los correos con la etiqueta 'mta_alerts' recibidos en los últimos 30 minutos,
parsea el cuerpo HTML de cada correo y extrae los siguientes campos:
  - Categoría de la alerta (retraso, cambio de servicio, reanudación, etc.)
  - Líneas de metro afectadas
  - Motivo del incidente
  - Ubicación aproximada
  - Fragmento de texto limpio (máximo 500 caracteres)

El resultado se sube como Parquet a MinIO en la ruta:
  grupo5/raw/official_alerts/DataFrame_Alertas_TiempoReal.parquet

Dependencias:
  - google-auth, google-auth-oauthlib, google-api-python-client : autenticación y acceso a Gmail
  - beautifulsoup4 : parseo del HTML del cuerpo del correo
  - pandas         : construcción del DataFrame de salida
  - src.common.minio_client.upload_df_parquet : subida a MinIO

Variables de entorno requeridas:
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY

Nota: La primera ejecución abre un flujo OAuth interactivo y guarda el token en
token.json. En ejecuciones posteriores el token se reutiliza o se refresca
automáticamente si ha expirado.
"""

import os
import base64
import re
import pandas as pd
from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from pathlib import Path

from src.common.minio_client import upload_df_parquet

# Permisos necesarios: solo lectura de Gmail.
# Si se modifican, hay que borrar token.json para regenerarlo.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"


def get_gmail_service():
    """
    Autentica con la API de Gmail y devuelve el objeto de servicio.

    Usa token.json si existe y es válido; si el token ha expirado lo refresca
    con el refresh_token; si no existe token alguno, lanza el flujo OAuth
    interactivo (abre el navegador) y guarda el nuevo token en token.json.
    """
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def parse_mta_body(html_content):
    """
    Parsea el cuerpo HTML de un correo de la MTA y extrae información estructurada.

    Extrae los siguientes campos:
      - lines    : líneas de metro afectadas (ej. 'A, C, E')
      - reason   : motivo del incidente (puertas, señales, clima, mantenimiento, etc.)
      - category : categoría del aviso (Delay, Service Change, Planned Work, etc.)
      - location : ubicación aproximada extraída del texto
      - snippet  : fragmento de texto limpio de hasta 500 caracteres

    Parámetros
    ----------
    html_content : Cadena con el HTML del cuerpo del correo.

    Devuelve
    --------
    Tupla (lines, reason, category, location, clean_text).
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Eliminar scripts y estilos del HTML
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    text = soup.get_text(separator=' ')
    clean_text = re.sub(r'\s+', ' ', text).strip()
    text_lower = clean_text.lower()

    # Categoría del aviso
    if any(word in text_lower for word in ["resumed", "regular service", "resolved"]):
        category = "Service Resumed"
    elif "preparing for" in text_lower and "storm" in text_lower:
        category = "Weather Prep"
    elif any(word in text_lower for word in ["delay", "held", "waiting", "slower"]):
        category = "Delay"
    elif any(word in text_lower for word in ["running local", "running express", "rerouted", "bypass"]):
        category = "Service Change"
    elif "planned work" in text_lower:
        category = "Planned Work"
    else:
        category = "Info/Other"

    # Líneas de metro afectadas
    line_pattern = r'\b([1-7]|A|B|C|D|E|F|G|J|L|M|N|Q|R|S|W|Z)\b'
    lines = sorted(list(set(re.findall(line_pattern, clean_text))))

    # Motivo específico del incidente
    reason = "Unknown"
    if "door" in text_lower:
        reason = "Mechanical (Doors)"
    elif "signal" in text_lower:
        reason = "Signal Problems"
    elif "person on the tracks" in text_lower or "ems" in text_lower:
        reason = "Medical/Police"
    elif "track" in text_lower and "work" in text_lower:
        reason = "Maintenance"
    elif "winter storm" in text_lower:
        reason = "Weather"

    # Ubicación: busca patrones del tipo "at/from/to/near + nombre"
    location = "Multiple/System-wide"
    loc_match = re.search(r'(?:at|from|to|near)\s+([A-Z][a-z0-9]+(?:\s[A-Z][a-z0-9]+)*)', clean_text)
    if loc_match:
        location = loc_match.group(1)

    return ", ".join(lines), reason, category, location, clean_text[:500]


def main():
    """
    Función principal: obtiene el servicio de Gmail, itera sobre los correos
    de los últimos 30 minutos con etiqueta 'mta_alerts', parsea cada uno y
    sube el DataFrame resultante a MinIO como Parquet.
    """
    service = get_gmail_service()
    data_log = []
    page_token = None

    # Solo se procesan correos de los últimos 30 minutos
    cutoff_utc = datetime.now(timezone.utc) - timedelta(minutes=30)

    print("Extrayendo correos de alertas MTA de los ultimos 30 minutos...")

    while True:
        results = service.users().messages().list(
            userId='me',
            q='label:mta_alerts newer_than:30m',
            pageToken=page_token
        ).execute()

        messages = results.get('messages', [])
        if not messages:
            break

        print(f"  Procesando lote de {len(messages)} correos...")

        for msg in messages:
            try:
                m = service.users().messages().get(
                    userId='me', id=msg['id'], format='full'
                ).execute()

                # internalDate viene en milisegundos epoch (UTC)
                timestamp_utc = pd.to_datetime(int(m['internalDate']), unit='ms', utc=True)
                timestamp_ny = timestamp_utc.tz_convert('America/New_York')

                # Descartamos correos fuera de la ventana de 30 minutos
                if timestamp_utc.to_pydatetime() < cutoff_utc:
                    continue

                # Extrae la parte HTML del correo (recursivo por si es multipart)
                def get_html_part(payload):
                    """Extrae recursivamente la parte text/html del payload de un mensaje de Gmail."""
                    if payload.get('mimeType') == 'text/html':
                        data = payload.get('body', {}).get('data')
                        if not data:
                            return None
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                    if 'parts' in payload:
                        for part in payload['parts']:
                            html = get_html_part(part)
                            if html:
                                return html
                    return None

                html_body = get_html_part(m.get('payload', {}))
                if not html_body:
                    continue

                lines, reason, category, location, clean_text = parse_mta_body(html_body)

                data_log.append({
                    'timestamp': timestamp_ny,
                    'category': category,
                    'lines': lines,
                    'reason': reason,
                    'location': location,
                    'text_snippet': clean_text,
                    'gmail_id': msg['id'],
                })

            except Exception:
                continue

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    if data_log:
        df = pd.DataFrame(data_log).sort_values(by='timestamp', ascending=False)

        # Eliminar duplicados por ID de correo
        df = df.drop_duplicates(subset=['gmail_id'])

        ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
        SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
        upload_df_parquet(ACCESS_KEY, SECRET_KEY, 'grupo5/raw/official_alerts/DataFrame_Alertas_TiempoReal.parquet', df)
        print(f"Dataset creado con {len(df)} filas (ultimos 30 minutos).")
    else:
        print("No se encontraron correos de alertas en los ultimos 30 minutos.")


if __name__ == '__main__':
    main()
