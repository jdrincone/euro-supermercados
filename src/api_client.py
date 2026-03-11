"""Cliente HTTP para la API de Euro Supermercados.

Encapsula autenticación, descarga de ventas y consulta de terceros.
Las credenciales se leen de variables de entorno ``API_USERNAME`` / ``API_PASSWORD``.
"""

import logging
import os
from datetime import timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://back-middleware.eurosupermercados.com"

_MAX_RETRIES = 3
_BACKOFF_FACTOR = 1  # 1s, 2s, 4s entre reintentos


def _session_with_retries() -> requests.Session:
    """Crea una sesión HTTP con reintentos automáticos y backoff exponencial."""
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_auth_token(
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Obtiene token de autenticación desde la API.

    Si no se pasan credenciales, se leen de las variables de entorno
    ``API_USERNAME`` y ``API_PASSWORD``.

    Raises:
        EnvironmentError: Si las variables de entorno no están definidas.
        requests.HTTPError: Si la autenticación falla.
    """
    username = username or os.environ.get("API_USERNAME")
    password = password or os.environ.get("API_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "Variables de entorno API_USERNAME y API_PASSWORD son requeridas. "
            "Revisa tu archivo .env."
        )
    session = _session_with_retries()
    resp = session.post(
        f"{BASE_URL}/api-auth/",
        json={"username": username, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["key"]


def fetch_sales(
    token: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[list[dict], list[str]]:
    """Descarga ventas día a día en el rango ``[start_date, end_date)``.

    Returns:
        Tupla (registros JSON, fechas con error).
    """
    url = f"{BASE_URL}/euro_nous_thirdparties/sales/get_sales/"
    headers = {"Authorization": f"key {token}", "Content-Type": "application/json"}

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    session = _session_with_retries()
    records: list[dict] = []
    failed_dates: list[str] = []
    current = start

    while current < end:
        date_str = current.strftime("%d/%m/%Y")
        logger.info("Consultando ventas: %s", date_str)
        try:
            resp = session.get(
                url,
                headers=headers,
                params={"date__date_sale": date_str},
                timeout=60,
            )
            resp.raise_for_status()
            records.extend(resp.json().get("data", []))
        except requests.RequestException as exc:
            logger.warning("Error descargando ventas del %s: %s", date_str, exc)
            failed_dates.append(date_str)
        current += timedelta(days=1)

    logger.info(
        "Ventas descargadas: %d registros, %d fallos", len(records), len(failed_dates)
    )
    return records, failed_dates


def fetch_third_parties(
    token: str,
    documents: list[str] | pd.Series,
    batch_size: int = 10,
) -> list[dict]:
    """Consulta información de terceros (clientes) en lotes.

    Args:
        token: Token de autenticación.
        documents: Lista de números de documento a consultar.
        batch_size: Tamaño de cada lote de consulta.

    Returns:
        Lista de diccionarios con datos de terceros.
    """
    url = f"{BASE_URL}/euro_nous_thirdparties/thirdparties/"
    headers = {"Authorization": f"key {token}", "Content-Type": "application/json"}

    session = _session_with_retries()
    docs = [str(d) for d in documents]
    all_records: list[dict] = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        logger.info(
            "Consultando terceros: %d-%d de %d", i + 1, i + len(batch), len(docs)
        )
        try:
            resp = session.get(
                url,
                headers=headers,
                params={"document__in": ",".join(batch)},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list):
                all_records.extend(data)
            else:
                logger.warning(
                    "Respuesta inesperada de la API para batch %d-%d",
                    i + 1,
                    i + len(batch),
                )
        except requests.RequestException as exc:
            logger.warning("Error en batch %d-%d: %s", i + 1, i + len(batch), exc)

    logger.info("Terceros obtenidos: %d registros", len(all_records))
    return all_records
