import pandas as pd
import yaml
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import joblib
import logging

from utils import obtener_token, obtener_ventas

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_backtesting(config_path):
    """
    Realiza backtesting comparando ventas reales con clientes de alta probabilidad por fecha,
    cliente a cliente.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Obtener Parámetros y Rutas ---
    data_params = config['data']
    train_params = config['train']
    eval_params = config['evaluate']
    model_params = config['model']
    backtest_params = config['backtesting']
    reports_params = config['reports']

    base_path = Path(data_params['base_path'])
    processed_path = base_path / data_params['processed_folder']
    model_path = Path(model_params['model_dir'])
    reports_path = Path(reports_params['reports_dir'])
    reports_path.mkdir(parents=True, exist_ok=True)

    features_file = processed_path / config['featurize']['output_file']
    calibrated_model_file = model_path / model_params['calibrated_model_name']
    threshold = eval_params['evaluation_threshold']

    username = os.environ.get('API_USERNAME')
    password = os.environ.get('API_PASSWORD')

    start_date_str = backtest_params['backtest_start_date']
    end_date_str = backtest_params['backtest_end_date']

    # --- 1. Descargar ventas del API para el rango de fechas ---
    logging.info(f"Descargando ventas del API desde {start_date_str} hasta {end_date_str}...")
    token = obtener_token(username, password)

    end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
    end_date_plus_one = end_date_dt + timedelta(days=1)
    end_date_str_plus_one = end_date_plus_one.strftime('%Y-%m-%d')

    nuevas_ventas, _ = obtener_ventas(token, start_date_str, end_date_str_plus_one)

    if not nuevas_ventas:
        logging.warning("No se descargaron ventas del API.")
        return

    df_nuevas_ventas = pd.DataFrame(nuevas_ventas)
    ventas_explotado = df_nuevas_ventas.explode('invoice_details').reset_index(drop=True)
    invoice_df = pd.json_normalize(ventas_explotado['invoice_details'])
    ventas_final = pd.concat([ventas_explotado.drop(columns=['invoice_details']).reset_index(drop=True),
                              invoice_df.reset_index(drop=True)], axis=1)
    ventas_final['date'] = pd.to_datetime(ventas_final['date_sale']).dt.normalize()
    ventas_final['client'] = ventas_final['identification_doct'].astype(str).str.strip()
    logging.info(f"Total de ventas descargadas: {len(ventas_final)}")
    logging.info(f"Ventas por fecha:\n{ventas_final.groupby(['date']).agg({'client': 'nunique'})}")

    # --- 2. Cargar el archivo daily (calendario de features) y obtener clientes de alta probabilidad ---
    logging.info(f"Cargando calendario de features desde: {features_file}")
    calendar = pd.read_parquet(features_file)
    calendar['date'] = pd.to_datetime(calendar['date']).dt.normalize()
    calendar['client'] = calendar['client'].astype(str)

    ventas_final = ventas_final[ventas_final['client'].isin(calendar['client'].unique())].copy()
    logging.info(f"Ventas filtradas por clientes en calendario: {len(ventas_final)}")

    logging.info(f"Cargando modelo calibrado desde: {calibrated_model_file}")
    calibrator = joblib.load(calibrated_model_file)
    features = train_params['features']
    calendar['prob'] = calibrator.predict_proba(calendar[features])[:, 1]
    high_prob_clients = calendar[calendar['prob'] >= threshold][['date', 'client', 'prob']].copy()
    logging.info(f"Total de clientes con alta probabilidad: {len(high_prob_clients)}")
    logging.info(f"Clientes de alta probabilidad por fecha (head):\n{high_prob_clients[['date', 'client', 'prob']].head()}")

    # --- 3. Realizar el backtesting por fecha, cliente a cliente ---
    start_date = pd.to_datetime(start_date_str).date()
    end_date = pd.to_datetime(end_date_str).date()
    backtest_dates = pd.date_range(start_date, end_date, freq="D")
    records = []

    logging.info("Iniciando backtesting por fecha, cliente a cliente...")
    for fecha in backtest_dates:
        fecha_dt = pd.to_datetime(fecha).normalize()
        logging.info(f"  Procesando fecha: {fecha_dt.date()}")

        # Ventas reales para la fecha actual
        ventas_hoy = ventas_final[ventas_final['date'] == fecha_dt][['client', 'date']].drop_duplicates()

        # Predicciones para la fecha actual
        predicciones_hoy = high_prob_clients[high_prob_clients['date'] == fecha_dt][['client', 'date']].drop_duplicates()

        # Unir los DataFrames por cliente y fecha para comparar
        merged_df = pd.merge(ventas_hoy, predicciones_hoy, on=['client', 'date'], how='outer', indicator=True)

        # Contar TP, FP, FN usando la columna _merge
        tp = len(merged_df[merged_df['_merge'] == 'both'])
        fp = len(merged_df[merged_df['_merge'] == 'right_only'])
        fn = len(merged_df[merged_df['_merge'] == 'left_only'])
        n_ventas_hoy = len(ventas_hoy)
        n_predicciones_hoy = len(predicciones_hoy)

        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        beta = 0.5
        f05 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if (precision + recall) > 0 else 0

        records.append({
            'fecha': fecha_dt.date(),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Clientes_Compraron': n_ventas_hoy,
            'Clientes_Predichos': n_predicciones_hoy,
            'Precision': precision,
            'Recall': recall,
            'F0.5-score': f05
        })

    # --- Guardar y Mostrar Resultados ---
    metrics_df = pd.DataFrame(records)
    output_file = reports_path / reports_params['backtesting_metrics_file']
    metrics_df.to_csv(output_file, index=False, float_format='%.4f')
    logging.info(f"Métricas de backtesting guardadas en: {output_file}")
    logging.info(f"Resumen de Métricas de Backtesting por Fecha (Cliente a Cliente):\n{metrics_df.to_string(index=False, float_format='%.4f')}")

    # Calcular y mostrar promedios
    mean_precision = metrics_df['Precision'].mean()
    mean_recall = metrics_df['Recall'].mean()
    mean_f05 = metrics_df['F0.5-score'].mean()
    print(f"\nPromedios del Periodo:")
    print(f" Precisión Promedio: {mean_precision:.4f}")
    print(f" Recall Promedio: {mean_recall:.4f}")
    print(f" F0.5 Promedio: {mean_f05:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    perform_backtesting(args.config)