# src/load_data.py
import pandas as pd
import yaml
from pathlib import Path
import argparse
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

from utils import obtener_token, obtener_ventas

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_clean(config_path):
    """Carga, limpia y guarda los datos de ventas desde la API."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    base_path = Path(data_params['base_path'])
    raw_path = base_path / data_params['raw_folder']
    processed_path = base_path / data_params['processed_folder']
    processed_path.mkdir(parents=True, exist_ok=True)

    username = os.environ.get('API_USERNAME')
    password = os.environ.get('API_PASSWORD')

    sales_file = raw_path / data_params['sales_file']
    months_to_fetch = data_params.get('months_to_fetch', 6)

    df_ventas_completo = pd.read_csv(sales_file, low_memory=False)
    df_ventas_completo['date_sale'] = pd.to_datetime(df_ventas_completo['date_sale'])

    fecha_last_train = df_ventas_completo['date_sale'].max().date()
    logging.info(f"Archivo de ventas existente cargado. Fecha máxima local: {fecha_last_train}")


    fecha_actual = datetime.now().date()
    logging.info(f"Obteniendo nuevas ventas desde la API desde {fecha_last_train} hasta {fecha_actual}")
    token = obtener_token(username, password)
    nuevas_ventas, _ = obtener_ventas(token, fecha_last_train, fecha_actual)

    if nuevas_ventas:
        df_nuevas_ventas = pd.DataFrame(nuevas_ventas)
        ventas_explotado = df_nuevas_ventas.explode('invoice_details').reset_index(drop=True)
        invoice_df = pd.json_normalize(ventas_explotado['invoice_details'])
        ventas_final = pd.concat([ventas_explotado.drop(columns=['invoice_details']).reset_index(drop=True),
                                  invoice_df.reset_index(drop=True)], axis=1)
        ventas_final['date_sale'] = pd.to_datetime(ventas_final['date_sale'])

        df_ventas_actualizado = pd.concat([df_ventas_completo, ventas_final], ignore_index=True)

        set_sales = ["ID", "date_sale", "identification_doct", "product"]
        df_ventas_actualizado.drop_duplicates(subset=set_sales, inplace=True)

        fecha_max = df_ventas_actualizado["date_sale"].max()
        fecha_min_work = fecha_max - pd.Timedelta(days=months_to_fetch * 30)
        df_procesar = df_ventas_actualizado[df_ventas_actualizado["date_sale"] >= fecha_min_work]

        df_procesar.to_csv(sales_file, index=False)

        logging.info(f"Archivo {sales_file} actualizado con {len(df_nuevas_ventas)} nuevas ventas.")
    else:
        logging.info("No hay nuevas ventas para actualizar.")
        df_procesar = df_ventas_completo

    # --- Data Cleaning ---
    # Renombrar y filtrar IDs de cliente
    df_procesar['id_client'] = df_procesar['identification_doct'].astype(str).str.strip()
    mask_digits = df_procesar["id_client"].str.isdigit().fillna(False)
    mask_zero = ~df_procesar["id_client"].str.startswith("0", na=False)
    df_procesar = df_procesar[mask_digits & mask_zero].copy()
    logging.info(f"Filas tras filtrar id_client no numéricos/cero inicial: {len(df_procesar)}")

    # Limpiar fechas, productos domicilio
    df_procesar['product'] = df_procesar['product'].astype(str).str.strip()
    df_procesar['date_sale'] = pd.to_datetime(df_procesar['date_sale'])
    df_procesar = df_procesar.dropna(subset=['date_sale'])
    df_procesar['date_sale'] = df_procesar['date_sale'].dt.normalize()
    df_procesar['domicilio_status'] = (
        df_procesar['domicilio_status']
        .astype(str)
        .str.strip()
        .str.lower()
        .eq('true')
    ).astype('boolean')


    # Select and save columns
    output_cols = config['load_data']['output_columns']
    df_out = df_procesar[output_cols].copy()
    output_file = processed_path / config['load_data']['output_file']
    df_out.to_parquet(output_file, index=False)
    logging.info(f"Datos guardados en: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    load_and_clean(args.config)