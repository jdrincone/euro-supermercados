#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/backtesting.py
==================

Compara, día a día, las compras reales contra los clientes con
probabilidad alta predicha por el modelo ya calibrado.

Para cada fecha del rango definido en `params.yaml` calcula:

* Verdaderos positivos (TP)  → clientes predichos **y** que compraron.
* Falsos positivos (FP)     → predichos pero sin compra.
* Falsos negativos (FN)     → compraron pero no predichos.
* Métricas: Precision, Recall, F0.5.

Resultados:
-----------
Se escribe un CSV con las métricas por fecha y se imprimen los promedios
del periodo.

Uso
----
$ python src/backtesting.py --config params.yaml
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv

from utils import obtener_token, obtener_ventas, read_yaml

# --------------------------------------------------------------------------- #
#  Logging & dotenv
# --------------------------------------------------------------------------- #
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------------------------------------------------------- #
#  Utilidades
# --------------------------------------------------------------------------- #


def download_sales(
    username: str,
    password: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Descarga ventas desde el API y las normaliza a DataFrame plano."""
    token = obtener_token(username, password)

    # API usa rango [start, end), por eso sumamos un día
    end_plus_one = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    raw_sales, _ = obtener_ventas(token, start, end_plus_one)

    if not raw_sales:
        logging.warning("No se descargaron ventas del API.")
        return pd.DataFrame()

    df_raw = pd.DataFrame(raw_sales)
    df_exp = df_raw.explode("invoice_details").reset_index(drop=True)
    df_details = pd.json_normalize(df_exp["invoice_details"])

    df = pd.concat(
        [df_exp.drop(columns="invoice_details").reset_index(drop=True), df_details],
        axis=1,
    )
    df["date"] = pd.to_datetime(df["date_sale"]).dt.normalize()
    df["client"] = df["identification_doct"].astype(str).str.strip()

    logging.info("🛒 Ventas descargadas: %s filas", len(df))
    return df


def load_calendar_and_model(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Any]:
    """Carga calendario de features y modelo calibrado."""
    processed = Path(cfg["data"]["base_path"]) / cfg["data"]["processed_folder"]
    calendar = pd.read_parquet(processed / cfg["featurize"]["output_file"])
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    calendar["client"] = calendar["client"].astype(str)

    model_path = (
        Path(cfg["model"]["model_dir"]) / cfg["model"]["calibrated_model_name"]
    )
    calibrator = joblib.load(model_path)
    logging.info("📅 Calendario: %s filas | Modelo cargado de %s", len(calendar), model_path)
    return calendar, calibrator


def predict_high_prob(
    calendar: pd.DataFrame, calibrator, features: List[str], thr: float
) -> pd.DataFrame:
    """Añade prob y devuelve filas con prob ≥ thr."""
    calendar = calendar.copy()
    calendar["prob"] = calibrator.predict_proba(calendar[features])[:, 1]
    hp = calendar[calendar["prob"] >= thr][["date", "client", "prob"]]
    logging.info("⭐ Clientes alta prob.: %s", len(hp))
    return hp


def daily_metrics(
    ventas: pd.DataFrame, preds: pd.DataFrame, dates: pd.DatetimeIndex
) -> List[Dict[str, Any]]:
    """Calcula TP/FP/FN y métricas para cada día."""
    records = []
    beta = 0.5

    for d in dates:
        d_norm = pd.to_datetime(d).normalize()
        v_hoy = ventas[ventas["date"] == d_norm][["client"]].drop_duplicates()
        p_hoy = preds[preds["date"] == d_norm][["client"]].drop_duplicates()

        merged = v_hoy.merge(p_hoy, on="client", how="outer", indicator=True)

        tp = (merged["_merge"] == "both").sum()
        fp = (merged["_merge"] == "right_only").sum()
        fn = (merged["_merge"] == "left_only").sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f05 = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if precision + recall
            else 0.0
        )

        records.append(
            {
                "fecha": d_norm.date(),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Clientes_Compraron": len(v_hoy),
                "Clientes_Predichos": len(p_hoy),
                "Precision": precision,
                "Recall": recall,
                "F0.5-score": f05,
            }
        )

    return records


def save_backtest(metrics: List[Dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    logging.info("💾 Métricas guardadas en %s", path)
    return df


# --------------------------------------------------------------------------- #
# Backtesting principal
# --------------------------------------------------------------------------- #
def perform_backtesting(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)

    # --- Rutas & params ---------------------------------------------------
    eval_thr = cfg["evaluate"]["evaluation_threshold"]
    start_date, end_date = cfg["backtesting"].values()
    dates_range = pd.date_range(start_date, end_date, freq="D")

    # --- Descarga ventas --------------------------------------------------
    sales = download_sales(
        os.environ["API_USERNAME"], os.environ["API_PASSWORD"], start_date, end_date
    )
    if sales.empty:
        return

    # --- Calendar & modelo -----------------------------------------------
    calendar, model = load_calendar_and_model(cfg)
    sales = sales[sales["client"].isin(calendar["client"].unique())].copy()

    high_prob = predict_high_prob(calendar, model, cfg["train"]["features"], eval_thr)

    # --- Métricas día a día ----------------------------------------------
    logging.info("⚖️  Calculando métricas diarias…")
    metrics = daily_metrics(sales, high_prob, dates_range)

    # --- Guardar resultados ----------------------------------------------
    reports_dir = Path(cfg["reports"]["reports_dir"])
    metrics_file = reports_dir / cfg["reports"]["backtesting_metrics_file"]

    df_metrics = save_backtest(metrics, metrics_file)
    logging.info("\n%s", df_metrics.to_string(index=False, float_format="%.4f"))

    # --- Promedios periodo -----------------------------------------------
    logging.info(
        "🏁 Promedios | Precision %.4f | Recall %.4f | F0.5 %.4f",
        df_metrics["Precision"].mean(),
        df_metrics["Recall"].mean(),
        df_metrics["F0.5-score"].mean(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting cliente a cliente.")
    parser.add_argument(
        "--config", default="params.yaml", help="Ruta al YAML de configuración."
    )
    args = parser.parse_args()
    perform_backtesting(args.config)
