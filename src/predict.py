#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicción de clientes con alta probabilidad de compra
y generación de recomendaciones de producto.
"""


import argparse
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import yaml
from dotenv import load_dotenv

from utils import obtener_token, obtener_terceros


load_dotenv()



def load_config(config_path: str) -> dict:
    """Carga el archivo YAML de configuración."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_and_features(config: dict):
    """Carga el modelo calibrado y el calendario de features."""
    data_params = config["data"]
    model_params = config["model"]

    processed_path = Path(data_params["base_path"]) / data_params["processed_folder"]
    model_path = Path(model_params["model_dir"])

    features_file = processed_path / config["featurize"]["output_file"]
    calibrated_model_file = model_path / model_params["calibrated_model_name"]

    try:
        print(f"Cargando modelo calibrado: {calibrated_model_file}")
        calibrator = joblib.load(calibrated_model_file)

        print(f"Cargando calendario de features: {features_file}")
        calendar = pd.read_parquet(features_file)
        calendar["date"] = pd.to_datetime(calendar["date"])
        calendar["client"] = calendar["client"].astype(str)

        return calibrator, calendar

    except FileNotFoundError as e:
        msg = (
            f"Archivo no encontrado: {e}. "
            "Asegúrate de haber ejecutado `dvc repro`."
        )
        raise FileNotFoundError(msg) from e


def process_prediction_dates(prediction_dates_str: list[str]) -> list[pd.Timestamp]:
    """Convierte strings de fecha a objetos datetime normalizados."""
    try:
        prediction_dates = [pd.to_datetime(d).normalize() for d in prediction_dates_str]
        print(
            "Fechas de predicción:",
            ", ".join(d.strftime("%Y-%m-%d") for d in prediction_dates),
        )
        return prediction_dates
    except ValueError as e:
        raise ValueError(
            f"Formato de fecha inválido (usa YYYY-MM-DD). Detalle: {e}"
        ) from e


def make_predictions(
    config: dict,
    calendar: pd.DataFrame,
    prediction_dates: list[pd.Timestamp],
    threshold: float,
) -> pd.DataFrame:
    """Genera probabilidades para las fechas indicadas y filtra por umbral."""
    train_params = config["train"]
    predict_df = calendar[calendar["date"].isin(prediction_dates)].copy()

    if predict_df.empty:
        print("No hay datos para las fechas especificadas.")
        print(
            f"Rango disponible: {calendar['date'].min().date()} "
            f"a {calendar['date'].max().date()}"
        )
        return pd.DataFrame()

    print(f"Realizando predicciones sobre {len(predict_df)} registros…")
    features = train_params["features"]

    calibrator, _ = load_model_and_features(config)  # Asegura modelo en memoria
    predict_df["prob"] = calibrator.predict_proba(predict_df[features])[:, 1]

    return predict_df[predict_df["prob"] >= threshold].copy()


def get_customer_info(predicted_clients: pd.Series) -> pd.DataFrame:
    """Obtiene info de contacto de clientes vía API propietaria."""
    username = os.environ["API_USERNAME"]
    password = os.environ["API_PASSWORD"]

    token = obtener_token(username, password)
    terceros = obtener_terceros(token, predicted_clients, batch_size=10)

    df_terceros = pd.DataFrame(terceros).rename(columns={"document": "client"})
    return df_terceros


def load_sales_and_products_data(config: dict) -> pd.DataFrame:
    """Carga ventas y catálogo de productos."""
    data_params = config["data"]
    raw_path = Path(data_params["base_path"]) / data_params["raw_folder"]


    df_ventas = pd.read_csv(
        raw_path / data_params["sales_file"],
        usecols=["identification_doct", "product", "date_sale"],
        low_memory=False,
    )
    df_ventas["client"] = df_ventas["identification_doct"].astype(str).str.strip()
    df_ventas = df_ventas[
        df_ventas["client"].str.isdigit() & ~df_ventas["client"].str.startswith("0")
    ].copy()

    df_ventas["product"] = df_ventas["product"].astype(str).str.strip()
    df_ventas["date_sale"] = pd.to_datetime(df_ventas["date_sale"]).dt.normalize()
    df_ventas = df_ventas.dropna(subset=["date_sale"])


    productos_df = (
        pd.read_csv(raw_path / data_params["products_file"], dtype={"codigo_unico": str})
        .rename(columns={"codigo_unico": "product"})
        .loc[:, ["product", "description"]]
        .drop_duplicates("product")
    )
    productos_df["product"] = productos_df["product"].str.strip()

    return df_ventas.merge(productos_df, on="product")


def filter_excluded_products(df_ventas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Excluye productos indicados en la configuración."""
    excluded = config["recommendations_clustering"].get(
        "excluded_product_descriptions", []
    )
    return df_ventas[~df_ventas["description"].isin(excluded)].copy()


def top_pct(group: pd.DataFrame, top_percentile: float = 0.75) -> pd.DataFrame:
    """Toma el 75 % superior de productos por frecuencia relativa."""
    k = math.ceil(len(group) * top_percentile)
    return group.head(k)


def generate_product_recommendations(
    df_ventas: pd.DataFrame,
    predicted_clients: pd.Series,
    recommendation_months: int = 1,
) -> pd.DataFrame:
    """Genera recomendaciones de producto basadas en compras recientes."""
    last_months_ago = datetime.now().date() - timedelta(days=recommendation_months * 30)

    recent_purchases = df_ventas[df_ventas["date_sale"].dt.date >= last_months_ago]
    recent_purchases_pred = recent_purchases[recent_purchases["client"].isin(predicted_clients)]

    client_daily = (
        recent_purchases_pred.groupby(["client", "date_sale", "description"])
        .size()
        .reset_index(name="cant_prod_dia")
    )

    client_stats = (
        client_daily.groupby(["client", "description"])
        .agg(avg_sale_period=("cant_prod_dia", "median"), dias_compro=("date_sale", "count"))
        .reset_index()
    )
    client_stats["avg_sale_period"] = client_stats["avg_sale_period"].astype(int)

    top_products = (
        client_stats.sort_values(["client", "dias_compro"], ascending=[True, False])
        .groupby("client", group_keys=False)
        .apply(top_pct)
        .reset_index(drop=True)
        .rename(columns={"description": "recommended_products"})
        .drop(columns="dias_compro")
    )

    return top_products



def save_predictions(pred_df: pd.DataFrame, path: Path) -> None:
    """Guarda dataframe de predicciones con formato estandarizado."""
    pred_df = pred_df.copy()
    pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")
    pred_df.sort_values(["date", "prob"], ascending=[True, False], inplace=True)

    pred_df.to_csv(path, index=False, float_format="%.4f")


def save_outputs(
    preds_recom_df: pd.DataFrame,
    preds_contact_df: pd.DataFrame,
    output_dir: str = "predictions",
    preds_file: str = "predictions_with_recommendations.csv",
    contacts_file: str = "predictions_client.csv",
) -> None:
    """
    Crea la carpeta de resultados (si no existe) y guarda dos archivos CSV.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    save_predictions(preds_recom_df, path / preds_file)
    preds_contact_df.to_csv(path / contacts_file, index=False)

    print(f"Archivos guardados en: {path.resolve()}")



def main(
    config_path: str,
    prediction_dates_str: list[str],
    threshold_override: float | None = None,
    output_filename: str = "predictions_with_recommendations.csv",
    recommendation_months: int = 1,
) -> None:
    """Orquesta el flujo completo de predicción + recomendaciones."""
    # --- Config y datos base ---------------------------------------------
    config = load_config(config_path)
    _, calendar = load_model_and_features(config)
    prediction_dates = process_prediction_dates(prediction_dates_str)

    eval_params = config["evaluate"]
    threshold = threshold_override or eval_params["evaluation_threshold"]
    print(f"Umbral de probabilidad: {threshold:.2f}")

    # --- Predicción de clientes -----------------------------------------
    predicted_clients_df = make_predictions(config, calendar, prediction_dates, threshold)
    if predicted_clients_df.empty:
        print("No se encontraron clientes con alta probabilidad.")
        return

    predicted_clients = predicted_clients_df["client"].unique()
    customer_info_df = get_customer_info(predicted_clients)

    # --- Datos de ventas + recomendaciones ------------------------------
    df_ventas = filter_excluded_products(load_sales_and_products_data(config), config)

    recommendations_df = generate_product_recommendations(
        df_ventas, predicted_clients, recommendation_months
    )


    preds_contact = predicted_clients_df[["date", "client", "prob"]].merge(
        customer_info_df[["name", "email", "phone", "telephone", "client"]], on="client"
    )

    preds_contact_recom = preds_contact.merge(
        recommendations_df, on="client", how="left"
    )

    save_outputs(
        preds_recom_df=preds_contact_recom,
        preds_contact_df=preds_contact,
        output_dir="predictions",
        preds_file=output_filename,
    )


#
# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predice clientes con alta probabilidad de compra y genera recomendaciones."
    )
    parser.add_argument(
        "--dates",
        required=True,
        nargs="+",
        help="Fechas de predicción en formato YYYY-MM-DD (separadas por espacio).",
    )
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Ruta al archivo de configuración YAML (default: params.yaml).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Umbral de probabilidad que sobrescribe el del YAML.",
    )
    parser.add_argument(
        "--output",
        default="predictions_with_recommendations.csv",
        help="Nombre del CSV principal de salida (default: predictions_with_recommendations.csv).",
    )
    parser.add_argument(
        "--recommendation_months",
        type=int,
        default=1,
        help="Meses hacia atrás a considerar para recomendaciones (default: 1).",
    )

    args = parser.parse_args()
    main(
        config_path=args.config,
        prediction_dates_str=args.dates,
        threshold_override=args.threshold,
        output_filename=args.output,
        recommendation_months=args.recommendation_months,
    )
