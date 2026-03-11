#!/usr/bin/env python3
"""Predicción de clientes con alta probabilidad de compra + recomendaciones.

Orquesta el flujo completo:
    1. Carga modelo calibrado y calendario de features.
    2. Predice clientes con probabilidad >= umbral para las fechas indicadas.
    3. Obtiene info de contacto del cliente vía API.
    4. Genera recomendaciones basadas en compras recientes.
    5. Guarda predicciones + recomendaciones en CSV.

Uso::

    python src/predict.py --dates 2025-05-08 2025-05-09 --threshold 0.5
"""

import argparse
import logging
import math
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import fetch_third_parties, get_auth_token
from config import load_config, processed_path
from data_io import load_calendar_features, load_calibrated_model, load_product_catalog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Predicción
# ---------------------------------------------------------------------------


def _parse_dates(dates_str: list[str]) -> list[pd.Timestamp]:
    """Convierte strings de fecha a Timestamps normalizados."""
    try:
        dates = [pd.to_datetime(d).normalize() for d in dates_str]
    except ValueError as exc:
        raise ValueError(f"Formato de fecha inválido (usar YYYY-MM-DD): {exc}") from exc
    logger.info(
        "Fechas de predicción: %s", ", ".join(d.strftime("%Y-%m-%d") for d in dates)
    )
    return dates


def _make_predictions(
    model,
    calendar: pd.DataFrame,
    features: list[str],
    dates: list[pd.Timestamp],
    threshold: float,
) -> pd.DataFrame:
    """Genera probabilidades y filtra por umbral."""
    df = calendar[calendar["date"].isin(dates)].copy()

    if df.empty:
        logger.warning(
            "Sin datos para fechas indicadas. Rango disponible: %s a %s",
            calendar["date"].min().date(),
            calendar["date"].max().date(),
        )
        return pd.DataFrame()

    df["prob"] = model.predict_proba(df[features])[:, 1]
    result = df[df["prob"] >= threshold].copy()
    logger.info("Predicciones: %d registros >= umbral %.2f", len(result), threshold)
    return result


# ---------------------------------------------------------------------------
#  Info de contacto
# ---------------------------------------------------------------------------


def _get_customer_info(clients: pd.Series) -> pd.DataFrame:
    """Obtiene datos de contacto de clientes vía API."""
    token = get_auth_token()
    records = fetch_third_parties(token, clients.tolist(), batch_size=10)
    return pd.DataFrame(records).rename(columns={"document": "client"})


# ---------------------------------------------------------------------------
#  Recomendaciones por historial reciente
# ---------------------------------------------------------------------------


def _load_sales_with_descriptions(cfg: dict) -> pd.DataFrame:
    """Carga ventas + catálogo de productos, excluyendo productos configurados."""
    proc = processed_path(cfg)
    data_cfg = cfg["data"]

    df = pd.read_parquet(proc / data_cfg["sales_file"])
    df = df.assign(
        product=df["product"].astype(str).str.strip(),
        date_sale=pd.to_datetime(df["date_sale"]).dt.normalize(),
        client=df["id_client"].astype(str),
    )
    df = df.dropna(subset=["date_sale"])

    catalog = load_product_catalog(proc / data_cfg["products_file"])
    df = df.merge(catalog, on="product")

    # Excluir productos configurados
    excluded = cfg.get("recommendations_clustering", {}).get(
        "excluded_product_descriptions", []
    )
    if excluded:
        df = df[~df["description"].isin(excluded)].copy()

    return df


def _generate_recommendations(
    df_sales: pd.DataFrame,
    predicted_clients: pd.Series,
    top_percentile: float,
    months: int = 1,
) -> pd.DataFrame:
    """Recomienda productos basados en compras recientes del cliente.

    Usa la fecha máxima del dataset como referencia (no datetime.now()).

    Args:
        df_sales: Ventas con descripción de producto.
        predicted_clients: IDs de clientes predichos.
        top_percentile: Fracción de productos a mantener (ej: 0.75 = top 75%).
        months: Meses hacia atrás a considerar.
    """
    max_date = df_sales["date_sale"].max()
    cutoff = max_date - timedelta(days=months * 30)

    recent = df_sales[
        (df_sales["date_sale"] >= cutoff) & (df_sales["client"].isin(predicted_clients))
    ]

    if recent.empty:
        return pd.DataFrame(
            columns=["client", "recommended_products", "avg_sale_period"]
        )

    # Frecuencia diaria por producto
    daily_counts = (
        recent.groupby(["client", "date_sale", "description"])
        .size()
        .reset_index(name="qty_day")
    )

    # Estadísticas por cliente-producto
    stats = (
        daily_counts.groupby(["client", "description"])
        .agg(avg_sale_period=("qty_day", "median"), dias_compro=("date_sale", "count"))
        .reset_index()
    )
    stats["avg_sale_period"] = stats["avg_sale_period"].astype(int)

    # Top percentile de productos por frecuencia
    top = (
        stats.sort_values(["client", "dias_compro"], ascending=[True, False])
        .groupby("client", group_keys=False)
        .apply(lambda g: g.head(math.ceil(len(g) * top_percentile)))
        .reset_index(drop=True)
        .rename(columns={"description": "recommended_products"})
        .drop(columns="dias_compro")
    )

    return top


# ---------------------------------------------------------------------------
#  Guardado
# ---------------------------------------------------------------------------


def _save_outputs(
    preds_recom: pd.DataFrame,
    preds_contact: pd.DataFrame,
    output_dir: Path,
    preds_file: str,
) -> None:
    """Guarda CSVs de predicciones + recomendaciones y contacto."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preds_recom.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", "prob"], ascending=[True, False])
    df.to_csv(output_dir / preds_file, index=False, float_format="%.4f")

    preds_contact.to_csv(output_dir / "predictions_client.csv", index=False)
    logger.info("Archivos guardados en: %s", output_dir.resolve())


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def main(
    config_path: str,
    prediction_dates_str: list[str],
    threshold_override: float | None = None,
    output_filename: str = "predictions_with_recommendations.csv",
    recommendation_months: int = 1,
) -> None:
    """Orquesta predicción + recomendaciones + info de contacto."""
    cfg = load_config(config_path)
    model = load_calibrated_model(cfg)
    calendar = load_calendar_features(cfg)
    dates = _parse_dates(prediction_dates_str)

    threshold = threshold_override or cfg["evaluate"]["evaluation_threshold"]
    features = cfg["train"]["features"]
    top_pct = cfg.get("recommendation_settings", {}).get("top_product_percentile", 0.75)
    logger.info(
        "Umbral de probabilidad: %.2f | Top productos: %.0f%%", threshold, top_pct * 100
    )

    # Predicción
    preds_df = _make_predictions(model, calendar, features, dates, threshold)
    if preds_df.empty:
        logger.info("Sin clientes con alta probabilidad.")
        return

    predicted_clients = preds_df["client"].unique()

    # Info de contacto
    customer_info = _get_customer_info(pd.Series(predicted_clients))

    # Recomendaciones
    sales = _load_sales_with_descriptions(cfg)
    recs = _generate_recommendations(
        sales,
        pd.Series(predicted_clients),
        top_pct,
        recommendation_months,
    )

    # Combinar resultados
    contact_cols = ["name", "email", "phone", "telephone", "client"]
    available_cols = [c for c in contact_cols if c in customer_info.columns]
    preds_contact = preds_df[["date", "client", "prob"]].merge(
        customer_info[available_cols],
        on="client",
    )
    preds_with_recs = preds_contact.merge(recs, on="client", how="left")

    _save_outputs(preds_with_recs, preds_contact, Path("predictions"), output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predice clientes con alta probabilidad de compra y genera recomendaciones."
    )
    parser.add_argument("--dates", required=True, nargs="+", help="Fechas YYYY-MM-DD.")
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml.")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Umbral de probabilidad."
    )
    parser.add_argument(
        "--output",
        default="predictions_with_recommendations.csv",
        help="CSV de salida.",
    )
    parser.add_argument(
        "--recommendation_months",
        type=int,
        default=1,
        help="Meses para recomendaciones.",
    )
    args = parser.parse_args()
    main(
        args.config, args.dates, args.threshold, args.output, args.recommendation_months
    )
