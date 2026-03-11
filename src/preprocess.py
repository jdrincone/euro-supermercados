#!/usr/bin/env python3
"""Etapa DVC ``preprocess`` — filtra clientes activos y genera datasets procesados.

Aplica reglas de consistencia en patrones de compra y produce:

1. ``filtered_agg_sales_for_rec.parquet`` — cliente-producto-día para recomendaciones.
2. ``daily.parquet`` — cliente-día para la etapa *featurize*.

Uso::

    python src/preprocess.py --config params.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client_filters import validate_client_ids
from config import load_config, processed_path
from data_io import load_parquet, save_parquet
from patterns import (
    apply_pattern_filters,
    compute_purchase_patterns,
    filter_recent_clients,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Agregaciones
# ---------------------------------------------------------------------------


def _aggregate_product_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas por (cliente, día, producto) → cantidad y monto."""
    return (
        df.rename(
            columns={
                "invoice_value_with_discount_and_without_iva": "amount_paid",
                "amount": "quantity",
            }
        )
        .groupby(["date_sale", "id_client", "product"])
        .agg(quantity=("quantity", "sum"), amount_paid=("amount_paid", "sum"))
        .reset_index()
        .sort_values(["id_client", "date_sale"])
    )


def _aggregate_daily_client(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen cliente-día para *featurize*.

    Renombra ``id_client`` → ``client`` y ``date_sale`` → ``date`` para
    mantener consistencia con el esquema downstream.
    """
    return (
        df.groupby(["id_client", "date_sale"], as_index=False)
        .agg(
            qty_tot=("quantity", "sum"),
            amount_tot=("amount_paid", "sum"),
            skus=("product", "nunique"),
        )
        .assign(purchased=1)
        .rename(columns={"id_client": "client", "date_sale": "date"})
    )


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def preprocess_sales(config_path: str | Path) -> None:
    """Filtra clientes por patrones de compra y genera datasets procesados."""
    cfg = load_config(config_path)
    prep_cfg = cfg["preprocess"]
    proc = processed_path(cfg)

    # Cargar ventas
    parquet_in = proc / cfg["load_data"]["output_file"]
    df = load_parquet(parquet_in, "Ventas brutas")

    # Validar IDs de clientes (cédulas colombianas)
    df = validate_client_ids(df, id_col="id_client")

    # Filtrar por actividad reciente
    df_recent = filter_recent_clients(df, cfg["data"].get("last_month_with_sale", 3))

    # Filtrar por patrones de compra
    patterns = compute_purchase_patterns(df_recent)
    valid_ids = apply_pattern_filters(
        patterns,
        min_purchase_days=prep_cfg["min_purchase_count"],
        max_median_days=prep_cfg["max_median_days_between"],
        max_std_days=prep_cfg["max_std_days_between"],
        min_products=prep_cfg["min_products_filter"],
    )
    df_filtered = df_recent[df_recent["id_client"].isin(valid_ids)].copy()
    logger.info("Filas tras filtros: %d", len(df_filtered))

    # Salida 1: cliente-producto-día para recomendaciones
    df_prod_daily = _aggregate_product_daily(df_filtered)
    save_parquet(
        df_prod_daily,
        proc / prep_cfg["recommendation_output_file"],
        "Recs cliente-producto-día",
    )

    # Salida 2: cliente-día para featurize
    df_daily = _aggregate_daily_client(df_prod_daily)
    save_parquet(df_daily, proc / prep_cfg["daily_output_file"], "Dataset cliente-día")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesa ventas para recomendaciones y featurize."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    preprocess_sales(args.config)
