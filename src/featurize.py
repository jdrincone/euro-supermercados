#!/usr/bin/env python3
"""Etapa DVC ``featurize`` — genera calendario cliente-día con features temporales.

A partir del dataset diario genera un calendario completo y añade:

- Variables de fecha: ``dow``, ``dom``, ``month``, ``is_weekend``, ``is_quincena``.
- Recencia: ``days_since_last`` (días desde la última compra).
- Ventanas móviles: ``cnt_1d``, ``cnt_3d``, ``cnt_7d``, etc.

Uso::

    python src/featurize.py --config params.yaml
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, processed_path
from data_io import load_parquet, save_parquet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Feature builders
# ---------------------------------------------------------------------------


def _build_full_calendar(daily: pd.DataFrame, max_future: int) -> pd.DataFrame:
    """Crea un MultiIndex (cliente, fecha) completo hasta ``max_future`` días después."""
    min_date, max_hist = daily["date"].min(), daily["date"].max()
    max_cal = max_hist + timedelta(days=max_future)
    logger.info("Calendario: %s -> %s", min_date.date(), max_cal.date())

    full_idx = pd.MultiIndex.from_product(
        [daily["client"].unique(), pd.date_range(min_date, max_cal, freq="D")],
        names=["client", "date"],
    )
    return (
        daily.set_index(["client", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )


def _add_date_features(df: pd.DataFrame, quincena_days: list[int]) -> pd.DataFrame:
    """Agrega variables temporales derivadas de la fecha."""
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_quincena"] = df["dom"].isin(quincena_days).astype(int)
    return df


def _add_recency(df: pd.DataFrame, fillna_days: int) -> pd.DataFrame:
    """Crea ``days_since_last`` por cliente (días desde la última compra PREVIA).

    Usa ``shift(1)`` para evitar fuga de información: la compra del día actual
    NO se usa para calcular la recencia de ese mismo día. Solo se consideran
    compras de días anteriores.

    Para clientes sin compras previas, se asigna ``fillna_days`` como valor
    de recencia para indicar "nunca compró recientemente".
    """
    df = df.sort_values(["client", "date"]).copy()

    # Marcar fechas donde hubo compra, luego shift(1) para excluir el día actual
    buy_dates = df["date"].where(df["purchased"].eq(1))
    prev_buy = (
        buy_dates.groupby(df["client"], group_keys=False)
        .shift(1)  # Solo compras ANTERIORES al día actual
        .groupby(df["client"], group_keys=False)
        .ffill()
    )

    min_allowed = df["date"].min() - timedelta(days=fillna_days + 1)
    prev_buy = prev_buy.fillna(min_allowed)

    df["days_since_last"] = (df["date"] - prev_buy).dt.days
    df.loc[prev_buy == min_allowed, "days_since_last"] = fillna_days
    return df


def _add_rolling_counts(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Crea columnas ``cnt_<w>d`` con suma de compras en ventana desplazada.

    Usa ``shift(1)`` para evitar fuga de información (lookahead bias):
    el conteo del día actual NO se incluye en la ventana.
    El primer día de cada cliente tiene ``min_periods=1`` → 0 por el shift.
    """
    df = df.copy()
    for w in windows:
        df[f"cnt_{w}d"] = df.groupby("client")["purchased"].transform(
            lambda x: x.rolling(w, min_periods=1).sum().shift(1).fillna(0)
        )
    return df


def _add_monetary_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Crea features de gasto y diversidad de productos en ventanas móviles.

    Usa ``shift(1)`` para evitar fuga: el gasto del día actual NO se incluye.
    """
    df = df.copy()
    grouped = df.groupby("client")
    for w in windows:
        df[f"avg_amount_{w}d"] = grouped["amount_tot"].transform(
            lambda x: x.rolling(w, min_periods=1).mean().shift(1).fillna(0)
        )
        df[f"avg_skus_{w}d"] = grouped["skus"].transform(
            lambda x: x.rolling(w, min_periods=1).mean().shift(1).fillna(0)
        )
    return df


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def create_features(config_path: str | Path) -> None:
    """Construye el calendario de features y lo guarda como Parquet."""
    cfg = load_config(config_path)
    feat_cfg = cfg["featurize"]
    proc = processed_path(cfg)

    daily = load_parquet(proc / cfg["preprocess"]["daily_output_file"], "Diario")

    calendar = _build_full_calendar(daily, feat_cfg["future_days_offset"])
    calendar = _add_date_features(calendar, feat_cfg["quincena_days"])
    calendar = _add_recency(calendar, feat_cfg["recency_fillna_days"])
    calendar = _add_rolling_counts(calendar, feat_cfg["rolling_windows"])
    calendar = _add_monetary_features(
        calendar, feat_cfg.get("monetary_windows", [7, 30])
    )

    logger.info("Features creadas: %s", list(calendar.columns))
    save_parquet(calendar, proc / feat_cfg["output_file"], "Calendario features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera calendario y features temporales."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    create_features(args.config)
