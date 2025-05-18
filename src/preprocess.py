#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/preprocess.py
=================

Filtra ventas de clientes “activos” (compraron en los últimos *n* meses),
aplica reglas de consistencia en patrones de compra y genera:

1. Un parquet (cliente-producto-día) para el motor de recomendaciones.
2. Un parquet (cliente-día) para la etapa *featurize* del modelo de churn / CLV.

Uso
----
$ python src/preprocess.py --config params.yaml
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils import read_yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --------------------------------------------------------------------------- #
#  Funciones auxiliares
# --------------------------------------------------------------------------- #



def months_ago(months: int) -> datetime.date:
    """Devuelve la fecha de *n* meses atrás (aprox 30 días c/u)."""
    return datetime.now().date() - timedelta(days=months * 30)


def load_sales_parquet(path: Path) -> pd.DataFrame:
    """Carga un parquet y reporta tamaño."""
    df = pd.read_parquet(path)
    logging.info("Datos cargados de %s | Filas: %s", path, len(df))
    return df


def filter_recent_clients(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Mantiene sólo clientes con compras en los últimos *months* meses."""
    cutoff = months_ago(months)
    recent_ids = df[df["date_sale"].dt.date >= cutoff]["id_client"].unique()
    logging.info("Clientes con compras en %s meses: %s", months, len(recent_ids))
    return df[df["id_client"].isin(recent_ids)].copy()


def compute_purchase_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de patrón de compra por cliente."""
    daily = (
        df.drop_duplicates(subset=["id_client", "date_sale"])[
            ["id_client", "date_sale", "product"]
        ]
        .sort_values(["id_client", "date_sale"])
    )
    daily["days_between"] = daily.groupby("id_client")["date_sale"].diff().dt.days

    patterns = (
        daily.groupby("id_client")
        .agg(
            last_date=("date_sale", "max"),
            first_date=("date_sale", "min"),
            purchase_days=("date_sale", "nunique"),
            product_distinct=("product", "nunique"),
            avg_days_between=("days_between", "mean"),
            median_days_between=("days_between", "median"),
            std_days_between=("days_between", "std"),
        )
        .reset_index()
    )
    return patterns


def apply_pattern_filters(
    patterns: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.Series:
    """Aplica reglas min/max definidas en YAML y devuelve ids válidos."""
    m = cfg["min_purchase_count"]
    med = cfg["max_median_days_between"]
    s = cfg["max_std_days_between"]
    p = cfg["min_products_filter"]

    filtered = (
        patterns[patterns["purchase_days"] >= m]
        .loc[lambda d: d["median_days_between"] < med]
        .loc[lambda d: d["std_days_between"] <= s]
        .loc[lambda d: d["product_distinct"] >= p]
    )

    logging.info(
        "Patrón de compra: %s → %s clientes válidos",
        len(patterns),
        len(filtered),
    )
    return filtered["id_client"]


def aggregate_product_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas por cliente, día y SKU."""
    df_ren = df.rename(
        columns={
            "invoice_value_with_discount_and_without_iva": "amount_paid",
            "amount": "quantity",
        }
    )
    agg = (
        df_ren.groupby(["date_sale", "id_client", "product"])
        .agg(quantity=("quantity", "sum"), amount_paid=("amount_paid", "sum"))
        .reset_index()
        .sort_values(["id_client", "date_sale"])
    )
    return agg


def aggregate_daily_client(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset resumido cliente-día para *featurize*."""
    daily = (
        df.groupby(["id_client", "date_sale"], as_index=False)
        .agg(
            qty_tot=("quantity", "sum"),
            amount_tot=("amount_paid", "sum"),
            skus=("product", "nunique"),
        )
        .assign(purchased=1)
        .rename(columns={"id_client": "client", "date_sale": "date"})
    )
    return daily


def save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info("%s guardado en %s | Filas: %s", label, path, len(df))


# --------------------------------------------------------------------------- #
#   Pre-processing principal
# --------------------------------------------------------------------------- #
def preprocess_sales(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)

    data_params = cfg["data"]
    prep_params = cfg["preprocess"]

    processed_path = Path(data_params["base_path"]) / data_params["processed_folder"]
    parquet_in = processed_path / cfg["load_data"]["output_file"]

    #  Cargar ventas
    df = load_sales_parquet(parquet_in)

    # Filtrar por actividad reciente
    df_recent = filter_recent_clients(df, data_params.get("last_month_with_sale", 3))

    # Filtrar por patrones de compra
    patterns = compute_purchase_patterns(df_recent)
    valid_ids = apply_pattern_filters(patterns, prep_params)
    df_filtered = df_recent[df_recent["id_client"].isin(valid_ids)]
    logging.info("Filas tras filtros finales: %s", len(df_filtered))

    # Agregación para recomendaciones
    df_prod_daily = aggregate_product_daily(df_filtered)
    rec_out = processed_path / prep_params["recommendation_output_file"]
    save_parquet(df_prod_daily, rec_out, "Recs cliente-producto-día")

    # Agregación cliente-día
    df_daily = aggregate_daily_client(df_prod_daily)
    daily_out = processed_path / prep_params["daily_output_file"]
    save_parquet(df_daily, daily_out, "Dataset cliente-día")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-procesa ventas para recomendaciones y featurize."
    )
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Ruta al archivo YAML de configuración (default: params.yaml)",
    )
    args = parser.parse_args()
    preprocess_sales(args.config)
