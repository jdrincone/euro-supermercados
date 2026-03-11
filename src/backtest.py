#!/usr/bin/env python3
"""Etapa DVC ``backtest`` — compara predicciones vs compras reales día a día.

Para cada fecha del rango configurado calcula TP, FP, FN y métricas
(Precision, Recall, F0.5). Los resultados se guardan en CSV.

Uso::

    python src/backtest.py --config params.yaml
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import fetch_sales, get_auth_token
from config import load_config, reports_dir
from data_io import load_calendar_features, load_calibrated_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _download_and_flatten_sales(start: str, end: str) -> pd.DataFrame:
    """Descarga ventas de la API y las aplana a un DataFrame con ``date`` y ``client``."""
    token = get_auth_token()
    # API usa rango [start, end), sumamos un día al final para incluir end_date
    end_plus = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    raw, _ = fetch_sales(token, start, end_plus)

    if not raw:
        logger.warning("Sin ventas descargadas de la API.")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    exploded = df.explode("invoice_details").reset_index(drop=True)
    details = pd.json_normalize(exploded["invoice_details"])
    flat = pd.concat(
        [exploded.drop(columns="invoice_details").reset_index(drop=True), details],
        axis=1,
    )
    flat["date"] = pd.to_datetime(flat["date_sale"]).dt.normalize()
    flat["client"] = flat["identification_doct"].astype(str).str.strip()
    logger.info("Ventas descargadas: %d filas", len(flat))
    return flat


def _predict_high_prob(
    calendar: pd.DataFrame,
    model: Any,
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Predice clientes con probabilidad >= umbral."""
    cal = calendar.copy()
    cal["prob"] = model.predict_proba(cal[features])[:, 1]
    high = cal[cal["prob"] >= threshold][["date", "client", "prob"]]
    logger.info("Clientes alta probabilidad: %d", len(high))
    return high


def _daily_metrics(
    ventas: pd.DataFrame,
    preds: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """Calcula TP/FP/FN y métricas por día."""
    beta = 0.5
    records: list[dict[str, Any]] = []

    for d in dates:
        d_norm = pd.to_datetime(d).normalize()
        # Normalizar ambos lados de la comparación para evitar desalineación
        v_day = ventas[ventas["date"].dt.normalize() == d_norm][
            ["client"]
        ].drop_duplicates()
        p_day = preds[preds["date"].dt.normalize() == d_norm][
            ["client"]
        ].drop_duplicates()

        merged = v_day.merge(p_day, on="client", how="outer", indicator=True)
        tp = (merged["_merge"] == "both").sum()
        fp = (merged["_merge"] == "right_only").sum()
        fn = (merged["_merge"] == "left_only").sum()

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f05 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if prec + rec else 0.0

        records.append(
            {
                "fecha": d_norm.date(),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Clientes_Compraron": len(v_day),
                "Clientes_Predichos": len(p_day),
                "Precision": prec,
                "Recall": rec,
                "F0.5-score": f05,
            }
        )

    return records


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def perform_backtesting(config_path: str | Path) -> None:
    """Ejecuta backtesting comparando predicciones con ventas reales.

    Las fechas se calculan dinámicamente: los últimos ``backtest_days`` días
    anteriores a hoy. Esto garantiza que el backtesting siempre evalúe datos
    recientes, sin importar cuándo se ejecute el pipeline.
    """
    cfg = load_config(config_path)

    thr = cfg["evaluate"]["evaluation_threshold"]
    bt_cfg = cfg["backtesting"]

    # Fechas dinámicas: últimos N días antes de hoy
    backtest_days = bt_cfg.get("backtest_days", 7)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=backtest_days)).strftime("%Y-%m-%d")
    dates_range = pd.date_range(start_date, end_date, freq="D")

    logger.info("Backtesting dinámico: %s -> %s (%d días)", start_date, end_date, backtest_days)

    # Descargar ventas reales
    sales = _download_and_flatten_sales(start_date, end_date)
    if sales.empty:
        return

    # Cargar calendario y modelo
    calendar = load_calendar_features(cfg)
    model = load_calibrated_model(cfg)

    # Filtrar ventas a clientes conocidos
    sales = sales[sales["client"].isin(calendar["client"].unique())].copy()

    # Predicciones
    high_prob = _predict_high_prob(calendar, model, cfg["train"]["features"], thr)

    # Métricas día a día
    logger.info("Calculando métricas diarias...")
    metrics = _daily_metrics(sales, high_prob, dates_range)

    # Guardar
    rpt_dir = reports_dir(cfg)
    metrics_file = rpt_dir / cfg["reports"]["backtesting_metrics_file"]
    rpt_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(metrics_file, index=False, float_format="%.4f")
    logger.info("Métricas guardadas: %s", metrics_file)
    logger.info("\n%s", df_metrics.to_string(index=False, float_format="%.4f"))
    logger.info(
        "Promedios | Precision %.4f | Recall %.4f | F0.5 %.4f",
        df_metrics["Precision"].mean(),
        df_metrics["Recall"].mean(),
        df_metrics["F0.5-score"].mean(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtesting de predicciones vs ventas reales."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    perform_backtesting(args.config)
