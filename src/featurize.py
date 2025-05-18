#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/featurize.py
================

Genera un calendario cliente-día completo y añade:

* Variables de fecha (dow, dom, mes, fin de semana, quincena).
* Recencia (`days_since_last`).
* Ventanas móviles con el conteo de compras (`cnt_7d`, `cnt_30d`, …).

El resultado se guarda como parquet y sirve de insumo para la fase de
entrenamiento del modelo de propensión de compra.

Uso
----
$ python src/featurize.py --config params.yaml
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import List


import pandas as pd
from utils import read_yaml

# --------------------------------------------------------------------------- #
#  Configuración global de logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------------------------------------------------------- #
#   Utilidades
# --------------------------------------------------------------------------- #



def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logging.info("Diario cargado: %s filas", len(df))
    return df


def build_full_calendar(daily: pd.DataFrame, max_future: int) -> pd.DataFrame:
    """Crea MultiIndex cliente-fecha completo hasta `max_future` días en el futuro."""
    min_date, max_hist = daily["date"].min(), daily["date"].max()
    max_cal = max_hist + timedelta(days=max_future)

    logging.info("Calendario: %s → %s", min_date.date(), max_cal.date())

    full_idx = pd.MultiIndex.from_product(
        [daily["client"].unique(), pd.date_range(min_date, max_cal, freq="D")],
        names=["client", "date"],
    )
    calendar = (
        daily.set_index(["client", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )
    return calendar


def add_date_features(df: pd.DataFrame, quincena_days: List[int]) -> pd.DataFrame:
    """Agrega variables temporales sencillas."""
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_quincena"] = df["dom"].isin(quincena_days).astype(int)
    return df


def add_recency(df: pd.DataFrame, fillna_days: int) -> pd.DataFrame:
    """Crea `days_since_last` por cliente."""
    df.sort_values(["client", "date"], inplace=True)

    # última compra previa
    prev_buy = (
        df.groupby("client", group_keys=False)["date"]
        .apply(lambda s: s.where(df.loc[s.index, "purchased"].eq(1)).ffill())
    )

    min_allowed = df["date"].min() - timedelta(days=fillna_days + 1)
    prev_buy = prev_buy.fillna(min_allowed)

    df["days_since_last"] = (df["date"] - prev_buy).dt.days
    df.loc[prev_buy == min_allowed, "days_since_last"] = fillna_days
    return df


def add_rolling_counts(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Crea columnas `cnt_<w>d` con la suma de compras ventana desplazada."""
    for w in windows:
        df[f"cnt_{w}d"] = (
            df.groupby("client")["purchased"]
            .transform(lambda x: x.rolling(w, min_periods=1).sum().shift(1).fillna(0))
        )
    return df


def save_calendar(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info("Calendario guardado en %s | Filas: %s", path, len(df))


# --------------------------------------------------------------------------- #
# Proceso principal
# --------------------------------------------------------------------------- #
def create_features(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)

    # --- Rutas ------------------------------------------------------------
    data_params = cfg["data"]
    feat_params = cfg["featurize"]

    processed_path = Path(data_params["base_path"]) / data_params["processed_folder"]
    daily_in = processed_path / cfg["preprocess"]["daily_output_file"]
    cal_out = processed_path / feat_params["output_file"]

    # --- Cargar diario ----------------------------------------------------
    daily = load_daily(daily_in)

    # --- Construir calendario completo ------------------------------------
    calendar = build_full_calendar(daily, feat_params["future_days_offset"])

    # --- Features ---------------------------------------------------------
    calendar = add_date_features(calendar, feat_params["quincena_days"])
    calendar = add_recency(calendar, feat_params["recency_fillna_days"])
    calendar = add_rolling_counts(calendar, feat_params["rolling_windows"])

    logging.info("Features creadas: %s", list(calendar.columns))

    # --- Guardar ----------------------------------------------------------
    save_calendar(calendar, cal_out)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera calendario y características temporales para el modelo."
    )
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Ruta al YAML de configuración (default: params.yaml)",
    )
    args = parser.parse_args()
    create_features(args.config)
