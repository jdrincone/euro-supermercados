#!/usr/bin/env python3
"""Etapa DVC ``load`` — sincroniza ventas locales con la API.

Si el archivo local ya contiene datos hasta hoy, no llama a la API.
En caso contrario, descarga las ventas nuevas, las limpia y actualiza
el Parquet local manteniendo solo la ventana temporal configurada.

Uso::

    python src/load_data.py --config params.yaml
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import fetch_sales, get_auth_token
from client_filters import validate_client_ids
from config import load_config, processed_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _load_local_sales(filepath: Path) -> pd.DataFrame:
    """Carga el Parquet local o devuelve DataFrame vacío."""
    if not filepath.exists():
        logger.warning("Archivo local no encontrado — se creará uno nuevo.")
        return pd.DataFrame()
    df = pd.read_parquet(filepath)
    logger.info("Datos locales cargados: %d filas", len(df))
    return df


def _clean_new_sales(raw: list[dict], output_cols: list[str]) -> pd.DataFrame:
    """Normaliza las ventas crudas de la API al esquema esperado."""
    if not raw:
        return pd.DataFrame(columns=output_cols)

    df = pd.DataFrame(raw)
    exploded = df.explode("invoice_details").reset_index(drop=True)
    details = pd.json_normalize(exploded["invoice_details"])
    merged = pd.concat(
        [exploded.drop(columns=["invoice_details"]).reset_index(drop=True), details],
        axis=1,
    )

    merged["date_sale"] = pd.to_datetime(merged["date_sale"]).dt.normalize()
    merged["id_client"] = merged["identification_doct"].astype(str).str.strip()
    merged["product"] = merged["product"].astype(str).str.strip()
    merged = merged.dropna(subset=["date_sale"])

    # Filtrar IDs inválidos (cédulas colombianas válidas)
    merged = validate_client_ids(merged, id_col="id_client")

    logger.info("Ventas nuevas limpias: %d filas", len(merged))
    return merged[output_cols].copy()


def _trim_to_window(
    df: pd.DataFrame,
    months: int,
    output_cols: list[str],
) -> pd.DataFrame:
    """Mantiene solo la ventana temporal y limpia tipos de texto."""
    max_date = df["date_sale"].max()
    cutoff = max_date - pd.Timedelta(days=months * 30)
    df = df[df["date_sale"] >= cutoff].copy()

    text_cols = ["product", "id_client"]
    df[text_cols] = df[text_cols].astype("string[pyarrow]").fillna(pd.NA)
    df = df.dropna(subset=text_cols).reset_index(drop=True)
    return df[output_cols]


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def load_and_clean(config_path: str | Path) -> None:
    """Sincroniza el histórico de ventas local con datos nuevos de la API.

    1. Lee configuración.
    2. Carga histórico local.
    3. Si faltan datos recientes, descarga de la API, limpia y guarda.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]

    output_file = processed_path(cfg) / cfg["load_data"]["output_file"]
    months_to_fetch = data_cfg.get("months_to_fetch", 6)
    output_cols = cfg["load_data"]["output_columns"]

    df_local = _load_local_sales(output_file)

    today = datetime.now().date()
    last_local = (
        df_local["date_sale"].max().date()
        if not df_local.empty
        else date.fromisoformat("2025-01-01")
    )
    logger.info("Fecha máxima local: %s | Hoy: %s", last_local, today)

    if last_local >= today:
        logger.info("Archivo local al día. Sin cambios.")
        return

    # Descargar ventas nuevas
    logger.info("Actualizando ventas (últimos %d meses)...", months_to_fetch)
    token = get_auth_token()
    raw_sales, _ = fetch_sales(token, str(last_local), str(today))

    if not raw_sales:
        logger.info("Sin ventas nuevas de la API.")
        return

    df_new = _clean_new_sales(raw_sales, output_cols)
    df_combined = pd.concat([df_local, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(
        subset=["date_sale", "id_client", "product"]
    )

    df_out = _trim_to_window(df_combined, months_to_fetch, output_cols)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_file, index=False, engine="pyarrow")
    logger.info("Datos guardados: %d filas en %s", len(df_out), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sincroniza ventas locales con la API."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    load_and_clean(args.config)
