# src/backtest.py
import pandas as pd
import yaml
from pathlib import Path
import argparse
import joblib
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
from datetime import timedelta

def perform_backtesting(config_path):
    """Realiza el backtesting usando datos futuros y el modelo calibrado."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    train_params = config['train']
    eval_params = config['evaluate']
    backtest_params = config['backtesting']
    model_params = config['model']
    reports_params = config['reports']

    base_path = Path(data_params['base_path'])
    raw_path = base_path / data_params['raw_folder']
    processed_path = base_path / data_params['processed_folder']
    model_path = Path(model_params['model_dir'])
    reports_path = Path(reports_params['reports_dir'])
    reports_path.mkdir(parents=True, exist_ok=True)

    # --- Cargar Datos Necesarios ---
    print("Cargando datos de backtesting...")
    backtest_file = raw_path / data_params['backtesting_file']
    # Usar las columnas definidas en load_data para consistencia, si aplica
    cols = config['load_data']['sales_columns']
    dtype_sales = config['load_data']['sales_dtype']
    ventas_back = pd.read_csv(
        backtest_file,
        usecols=cols,
        dtype=dtype_sales
    ).rename(columns={'identification_doct': 'client'})

    ventas_back['client'] = ventas_back['client'].str.strip().astype(str) # Asegurar tipo str
    ventas_back['date_sale'] = pd.to_datetime(ventas_back['date_sale'], errors='coerce', dayfirst=True)
    ventas_back.dropna(subset=['date_sale'], inplace=True)
    ventas_back['date_sale'] = ventas_back['date_sale'].dt.normalize()
    print(f"  Datos crudos backtesting: {len(ventas_back)} filas")

    print("Cargando calendario de features y modelo calibrado...")
    features_file = processed_path / config['featurize']['output_file']
    calendar = pd.read_parquet(features_file)
    calendar['client'] = calendar['client'].astype(str) # Asegurar tipo str

    calibrated_model_file = model_path / model_params['calibrated_model_name']
    calibrator = joblib.load(calibrated_model_file)

    # Filtrar ventas_back para incluir solo clientes del calendario
    clients_in_calendar = calendar['client'].unique()
    ventas_back_filtered = ventas_back[ventas_back['client'].isin(clients_in_calendar)].copy()
    print(f"  Datos backtesting filtrados por cliente: {len(ventas_back_filtered)} filas")

    # Obtener compras reales por día/cliente para el periodo de backtesting
    clientes_by_day_actual = ventas_back_filtered[['client', 'date_sale']].drop_duplicates()
    print(f"  Compras reales únicas (cliente-día) en backtesting: {len(clientes_by_day_actual)}")

    # --- Bucle de Backtesting ---
    features = train_params['features']
    threshold = eval_params['evaluation_threshold']
    start_date = pd.to_datetime(backtest_params['backtest_start_date'])
    end_date = pd.to_datetime(backtest_params['backtest_end_date'])
    backtest_dates = pd.date_range(start_date, end_date, freq="D")

    print(f"Iniciando backtesting desde {start_date.date()} hasta {end_date.date()} con umbral {threshold:.2f}")
    records = []
    for fecha in backtest_dates:
        print(f"  Procesando fecha: {fecha.date()}")

        # Obtener predicciones para la fecha actual del calendario
        future_df = calendar[calendar['date'] == fecha].copy()
        if future_df.empty:
            print(f"    No hay datos en calendario para {fecha.date()}, saltando.")
            records.append({
                'fecha_comparacion': fecha, 'TP': 0, 'FP': 0, 'FN': 0,
                'Precision': 0, 'Recall': 0, 'F0.5-score': 0, 'Clientes_Reales': 0, 'Clientes_Predichos': 0
            })
            continue

        future_df['prob'] = calibrator.predict_proba(future_df[features])[:, 1]
        high_pred = future_df[future_df['prob'] >= threshold][['client', 'date', 'prob']] # Clientes predichos como compradores
        n_predichos = len(high_pred)

        # Obtener compras reales para la fecha actual
        clientes_actual_fecha = clientes_by_day_actual[clientes_by_day_actual["date_sale"] == fecha][['client', 'date_sale']]
        n_reales = len(clientes_actual_fecha)

        # Si no hubo compras reales ese día
        if n_reales == 0:
            tp = 0
            fp = n_predichos # Todos los predichos son falsos positivos
            fn = 0 # No hubo compras reales que fallamos
            print(f"    No hubo compras reales. Predichos: {n_predichos} (FP)")
        # Si no hubo predicciones ese día
        elif n_predichos == 0:
            tp = 0
            fp = 0
            fn = n_reales # Todas las compras reales son falsos negativos
            print(f"    No hubo predicciones. Reales: {n_reales} (FN)")
        # Si hubo ambos, calcular métricas
        else:
            # Comparar predichos con reales
            rev_back = pd.merge(
                clientes_actual_fecha.rename(columns={'date_sale': 'date'}), # Renombrar para merge
                high_pred,
                on=["client", "date"], # Merge por cliente y fecha
                how='outer',
                indicator=True
            )

            counts = rev_back["_merge"].value_counts()
            tp = counts.get("both", 0)         # Predicho y Compró
            fp = counts.get("right_only", 0)   # Predicho y No Compró
            fn = counts.get("left_only", 0)    # No Predicho y Compró
            # tn no se calcula directamente aquí, pero no se necesita para Precision/Recall/F

            print(f"    TP: {tp}, FP: {fp}, FN: {fn}. Reales: {n_reales}, Predichos: {n_predichos}")


        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

        records.append({
            'fecha_comparacion': fecha,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Clientes_Reales': n_reales,
            'Clientes_Predichos': n_predichos,
            'Precision': precision,
            'Recall': recall,
            'F0.5-score': f05
        })

    # --- Guardar y Mostrar Resultados ---
    metrics_df = pd.DataFrame(records)
    output_file = reports_path / reports_params['backtesting_metrics_file']
    metrics_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\nMétricas de backtesting guardadas en: {output_file}")

    print("\nResumen de Métricas de Backtesting:")
    print(metrics_df.to_string(index=False, float_format='%.4f')) # Mostrar tabla completa

    # Calcular y mostrar promedios
    mean_precision = metrics_df['Precision'].mean()
    mean_recall = metrics_df['Recall'].mean()
    mean_f05 = metrics_df['F0.5-score'].mean()
    print(f"\nPromedios del Periodo:")
    print(f"  Precisión Promedio: {mean_precision:.4f}")
    print(f"  Recall Promedio:    {mean_recall:.4f}")
    print(f"  F0.5 Promedio:      {mean_f05:.4f}")

    # Opcional: Añadir promedios al metrics.json principal
    # try:
    #     with open(reports_path / reports_params['metrics_file'], 'r') as f:
    #         main_metrics = json.load(f)
    # except FileNotFoundError:
    #     main_metrics = {}
    # main_metrics['backtest_mean_precision'] = mean_precision
    # main_metrics['backtest_mean_recall'] = mean_recall
    # main_metrics['backtest_mean_f05'] = mean_f05
    # with open(reports_path / reports_params['metrics_file'], 'w') as f:
    #     json.dump(main_metrics, f, indent=4)
    # print("Promedios de backtesting añadidos a reports/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    perform_backtesting(args.config)