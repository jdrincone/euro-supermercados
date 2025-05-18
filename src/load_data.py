#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/load_data.py
================

Sincroniza el histórico de ventas local con los registros más recientes
obtenidos vía API.  Si el archivo local ya contiene datos hasta la fecha
actual, no se realiza ninguna llamada adicional.

Uso
----
$ python src/load_data.py --config params.yaml
"""

import argparse
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils import obtener_token, read_yaml, obtener_ventas



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



def build_paths(data_params: Dict[str, Any]) -> tuple[Path, Path]:
    """
    Construye rutas base para archivos procesados.

    Returns
    -------
    processed_path : Path
        Ruta al directorio de datos procesados.
    output_file    : Path
        Archivo parquet con ventas ya consolidadas.
    """
    base_path = Path(data_params["base_path"])
    processed_path = base_path / data_params["processed_folder"]
    output_file = processed_path / data_params["output_file"]
    return processed_path, output_file


def load_local_sales(filepath: Path) -> pd.DataFrame:
    """Carga el parquet local, o devuelve DataFrame vacío si no existe."""
    if not filepath.exists():
        logging.warning("No se encontró archivo local, se creará uno nuevo.")
        return pd.DataFrame()

    df = pd.read_parquet(filepath)
    logging.info("Datos locales cargados: %s filas", len(df))
    return df



def needs_update(last_date: date, today: date) -> bool:
    """Indica si el histórico requiere actualización."""
    return last_date < today

def update_data_with_new_sales(df_ventas_completo, months_to_fetch, output_cols):
    fecha_last_train = df_ventas_completo['date_sale'].max().date()
    logging.info(f"Archivo de ventas existente cargado. Fecha máxima local: {fecha_last_train}")

    fecha_actual = datetime.now().date()
    logging.info(f"Obteniendo nuevas ventas desde la API desde {fecha_last_train} hasta {fecha_actual}")
    username = os.environ.get('API_USERNAME')
    password = os.environ.get('API_PASSWORD')
    token = obtener_token(username, password)
    nuevas_ventas, _ = obtener_ventas(token, fecha_last_train, fecha_actual)

    if nuevas_ventas:
        df_nuevas_ventas = pd.DataFrame(nuevas_ventas)
        ventas_explotado = df_nuevas_ventas.explode('invoice_details').reset_index(drop=True)
        invoice_df = pd.json_normalize(ventas_explotado['invoice_details'])
        ventas_final = pd.concat([ventas_explotado.drop(columns=['invoice_details']).reset_index(drop=True),
                                  invoice_df.reset_index(drop=True)], axis=1)
        ventas_final['date_sale'] = pd.to_datetime(ventas_final['date_sale'])

        # --- Data Cleaning ---
        ventas_final['id_client'] = ventas_final['identification_doct'].astype(str).str.strip()
        mask_digits = ventas_final["id_client"].str.isdigit().fillna(False)
        mask_zero = ~ventas_final["id_client"].str.startswith("0", na=False)
        df_procesar = ventas_final[mask_digits & mask_zero].copy()
        logging.info(f"Filas tras filtrar id_client no numéricos/cero inicial: {len(df_procesar)}")

        # Limpiar fechas, productos domicilio
        df_procesar['product'] = df_procesar['product'].astype(str).str.strip()
        df_procesar['date_sale'] = pd.to_datetime(df_procesar['date_sale'])
        df_procesar = df_procesar.dropna(subset=['date_sale'])
        df_procesar['date_sale'] = df_procesar['date_sale'].dt.normalize()

        df_ventas_actualizado = pd.concat([df_ventas_completo, ventas_final], ignore_index=True)

        set_sales = ["date_sale", "id_client", "product"]
        df_ventas_actualizado.drop_duplicates(subset=set_sales, inplace=True)
        fecha_max = df_ventas_actualizado["date_sale"].max()
        fecha_min_work = fecha_max - pd.Timedelta(days=months_to_fetch * 30)

        df_procesar = df_ventas_actualizado[df_ventas_actualizado["date_sale"] >= fecha_min_work]
        df_out = df_procesar[output_cols].copy()
        text_cols = ["product", "id_client"]
        df_out[text_cols] = (
            df_out[text_cols]
            .astype("string[pyarrow]")
            .fillna(pd.NA)
        )
        df_out = df_out.dropna(subset=text_cols).reset_index(drop=True)

        return df_out


def save_sales(df: pd.DataFrame, filepath: Path) -> None:
    """Guarda el DataFrame en parquet creando carpetas si es necesario."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logging.info("Datos guardados en %s", filepath.resolve())



def load_and_clean(config_path: str | Path) -> None:
    """
    Paso a paso
    -----------
    1. Lee configuración.
    2. Carga histórico local en parquet.
    3. Si faltan datos recientes, llama `update_data_with_new_sales`
       y guarda el dataset actualizado.
    """
    cfg = read_yaml(config_path)
    data_params = cfg["data"]
    processed_path, output_file = build_paths(data_params)
    months_to_fetch = data_params.get("months_to_fetch", 6)
    output_cols = cfg["load_data"]["output_columns"]

    df_sales = load_local_sales(output_file)

    today = datetime.now().date()
    last_local_date = (
        df_sales["date_sale"].max().date()
        if not df_sales.empty
        else date.fromisoformat("2025-01-01")
    )
    logging.info("Fecha máxima local: %s | Hoy: %s", last_local_date, today)

    if needs_update(last_local_date, today):
        logging.info("Actualizando ventas (últimos %s meses)…", months_to_fetch)
        df_sales = update_data_with_new_sales(
            df_sales, months_to_fetch, output_cols
        )
        save_sales(df_sales, output_file)
    else:
        logging.info("El archivo local ya está al día. No se realizaron cambios.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sincroniza el histórico de ventas local con nuevos datos."
    )
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Ruta al archivo YAML de configuración (default: params.yaml).",
    )
    args = parser.parse_args()
    load_and_clean(args.config)
