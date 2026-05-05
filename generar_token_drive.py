"""
Genera el archivo token_drive.json mediante el flujo OAuth 2.0 interactivo del usuario.

Este script debe ejecutarse una sola vez de forma local para autenticar la cuenta de
Google y guardar las credenciales en disco. El token resultante es utilizado por otros
módulos del proyecto (upload_daily_data.py, upload_realtime_window.py, etc.) para
acceder a Google Drive sin requerir intervención manual posterior.

Dependencias:
    - google-auth-oauthlib
    - google-auth
    - google-api-python-client

Requisito previo:
    Descarga credentials.json desde Google Cloud Console:
    APIs & Services → Credentials → OAuth 2.0 Client IDs → Desktop App → Download JSON
    Colócalo en la raíz del proyecto con el nombre credentials.json

Uso:
    uv run python generar_token_drive.py

El token resultante se guarda en:
    src/ETL/alertas_oficiales_tiempo_real/token_drive.json
"""
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive"]

CREDENTIALS_PATH = Path("credentials.json")
TOKEN_PATH = Path("src/ETL/alertas_oficiales_tiempo_real/token_drive.json")


def main():
    """
    Flujo principal de autenticación OAuth 2.0.

    Si el token ya existe y es válido, no hace nada adicional. Si ha expirado
    pero tiene refresh_token, lo refresca automáticamente. Si no existe, abre
    el navegador para completar el flujo de autorización interactivo y guarda
    el token resultante en TOKEN_PATH.

    Lanza FileNotFoundError si credentials.json no existe en la raíz del proyecto.
    """
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            "No se encuentra credentials.json en la raíz del proyecto.\n"
            "Descárgalo desde Google Cloud Console:\n"
            "  APIs & Services → Credentials → OAuth 2.0 Client IDs → Desktop App → Download JSON\n"
            f"  y renómbralo a {CREDENTIALS_PATH}"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("Token refrescado.")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
            print("Autorización completada.")

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Token guardado en: {TOKEN_PATH}")


if __name__ == "__main__":
    main()
