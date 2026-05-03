"""
Extraccion de alertas oficiales de la MTA en tiempo real via Gmail API.

Lee los correos con etiqueta 'mta_alerts' recibidos en los ultimos 30 minutos,
parsea el HTML del cuerpo y extrae: categoria, lineas afectadas, motivo,
ubicacion y un fragmento de texto.

Resultado: mta_dataset.csv con las alertas procesadas.
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
    """Autenticacion con Gmail API. Usa token.json si existe;
    si no, lanza el flujo OAuth interactivo y lo guarda."""
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
    '''
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            BASE_DIR = Path(__file__).resolve().parent
            flow = InstalledAppFlow.from_client_secrets_file(str(BASE_DIR / "credentials.json"), SCOPES)
            #flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    '''
    return build('gmail', 'v1', credentials=creds)


def parse_mta_body(html_content):
    """Parsea el cuerpo HTML de un correo de la MTA y extrae:
    - lineas afectadas (ej: 'A, C, E')
    - motivo del aviso (puertas, senales, clima, etc.)
    - categoria (retraso, cambio de servicio, etc.)
    - ubicacion aproximada
    - fragmento de texto limpio (max 500 chars)
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Eliminar scripts y estilos del HTML
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    text = soup.get_text(separator=' ')
    clean_text = re.sub(r'\s+', ' ', text).strip()
    text_lower = clean_text.lower()

    # 1) Categoria del aviso
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

    # 2) Lineas de metro afectadas
    line_pattern = r'\b([1-7]|A|B|C|D|E|F|G|J|L|M|N|Q|R|S|W|Z)\b'
    lines = sorted(list(set(re.findall(line_pattern, clean_text))))

    # 3) Motivo especifico
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

    # 4) Ubicacion (busca patrones tipo "at/from/to/near + nombre")
    location = "Multiple/System-wide"
    loc_match = re.search(r'(?:at|from|to|near)\s+([A-Z][a-z0-9]+(?:\s[A-Z][a-z0-9]+)*)', clean_text)
    if loc_match:
        location = loc_match.group(1)

    return ", ".join(lines), reason, category, location, clean_text[:500]


def main():
    service = get_gmail_service()
    data_log = []
    page_token = None

    # Punto de corte: solo correos de los ultimos 30 minutos
    cutoff_utc = datetime.now(timezone.utc) - timedelta(minutes=30)

    print("Extrayendo correos de alertas MTA de los ultimos 30 minutos...")

    while True:
        # Buscar correos con etiqueta mta_alerts de los ultimos 30 min
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
                # Descartamos si esta fuera de la ventana de 30 min
                if timestamp_utc.to_pydatetime() < cutoff_utc:
                    continue

                # Extraer la parte HTML del correo (recursivo por si es multipart)
                def get_html_part(payload):
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

                # Parsear el HTML y extraer campos
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

    # Construir DataFrame y exportar
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
