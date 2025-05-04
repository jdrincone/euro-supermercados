# src/predict.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
import joblib
from datetime import datetime
import os # Importar os para manejo de directorios

def predict_high_probability_clients(config_path, prediction_dates_str, threshold_override=None, output_filename='predictions.csv'): # Cambiado a output_filename
    """
    Predice clientes con alta probabilidad de compra para fechas dadas
    y añade su información de contacto. Guarda en la carpeta 'predictions/'.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Obtener Parámetros y Rutas ---
    data_params = config['data']
    train_params = config['train']
    eval_params = config['evaluate']
    model_params = config['model']

    base_path = Path(data_params['base_path'])
    raw_path = base_path / data_params['raw_folder']
    processed_path = base_path / data_params['processed_folder']
    model_path = Path(model_params['model_dir'])

    # --- Directorio de Salida para Predicciones ---
    predictions_dir = Path('predictions') # Nombre de la carpeta de salida
    predictions_dir.mkdir(parents=True, exist_ok=True) # Crear carpeta si no existe
    output_file_path = predictions_dir / output_filename # Ruta completa del archivo
    # ---------------------------------------------

    features_file = processed_path / config['featurize']['output_file']
    calibrated_model_file = model_path / model_params['calibrated_model_name']
    terceros_file = raw_path / data_params['terceros_file']

    # Usar umbral de params.yaml a menos que se especifique uno
    threshold = threshold_override if threshold_override is not None else eval_params['evaluation_threshold']
    print(f"Usando umbral de probabilidad: {threshold:.2f}")

    # --- Cargar Modelo, Features y Datos de Contacto ---
    try:
        print(f"Cargando modelo calibrado desde: {calibrated_model_file}")
        calibrator = joblib.load(calibrated_model_file)

        print(f"Cargando calendario de features desde: {features_file}")
        calendar = pd.read_parquet(features_file)
        calendar['date'] = pd.to_datetime(calendar['date']) # Asegurar tipo datetime
        calendar['client'] = calendar['client'].astype(str) # Asegurar tipo str

        print(f"Cargando información de contacto desde: {terceros_file}")
        terceros = pd.read_csv(
            terceros_file,
            converters={"document": str} # Leer como string
        )
        terceros = terceros.drop_duplicates("document")
        contact_cols = ['document', 'name', 'email', 'telephone'] # Ajustar si las columnas se llaman diferente
        terceros = terceros.loc[:, contact_cols]
        terceros['document'] = terceros['document'].str.strip()
        terceros.rename(columns={'document': 'client'}, inplace=True) # Renombrar para merge
        terceros['client'] = terceros['client'].astype(str) # Asegurar tipo str

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}. Asegúrate de haber ejecutado el pipeline DVC (`dvc repro`) primero.")
        return
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        return

    # --- Procesar Fechas y Realizar Predicciones ---
    try:
        prediction_dates = [pd.to_datetime(d).normalize() for d in prediction_dates_str]
        print(f"Fechas para predicción: {[d.strftime('%Y-%m-%d') for d in prediction_dates]}")
    except ValueError as e:
        print(f"Error: Formato de fecha inválido. Usa YYYY-MM-DD. Detalle: {e}")
        return

    # Filtrar calendario para las fechas solicitadas
    predict_df = calendar[calendar['date'].isin(prediction_dates)].copy()

    if predict_df.empty:
        print("Error: No se encontraron datos de features para las fechas especificadas en el calendario procesado.")
        print(f"Rango de fechas en calendario: {calendar['date'].min().date()} a {calendar['date'].max().date()}")
        return

    print(f"Realizando predicciones para {len(predict_df)} registros cliente-día...")
    features = train_params['features']
    try:
        predict_df['prob'] = calibrator.predict_proba(predict_df[features])[:, 1]
    except ValueError as e:
         print(f"Error durante la predicción. ¿Faltan features o hay tipos de datos incorrectos? Detalle: {e}")
         print("Features esperadas:", features)
         print("Features encontradas:", predict_df[features].columns.tolist())
         print("Tipos de datos encontrados:\n", predict_df[features].dtypes)
         return
    except Exception as e:
        print(f"Error inesperado durante la predicción: {e}")
        return

    # Filtrar por umbral
    high_prob_clients = predict_df[predict_df['prob'] >= threshold].copy()
    print(f"Clientes encontrados con probabilidad >= {threshold:.2f}: {len(high_prob_clients)}")

    if high_prob_clients.empty:
        print("No se encontraron clientes con probabilidad de compra por encima del umbral para las fechas dadas.")
        return

    # --- Añadir Información de Contacto y Guardar ---
    print("Añadiendo información de contacto...")
    results = pd.merge(
        high_prob_clients[['date', 'client', 'prob']],
        terceros,
        on='client',
        how='left' # Mantener todas las predicciones, incluso si no hay contacto
    )

    # Reordenar y formatear
    results['date'] = results['date'].dt.strftime('%Y-%m-%d')
    results = results.sort_values(['date', 'prob'], ascending=[True, False])
    final_cols = ['date', 'client', 'name', 'email', 'telephone', 'prob']
    # Asegurarse que todas las columnas existan, rellenando NAs si alguna de contacto falta
    for col in final_cols:
        if col not in results.columns:
            results[col] = np.nan
    results = results[final_cols]

    # Guardar en CSV dentro de la carpeta predictions/
    try:
        results.to_csv(output_file_path, index=False, float_format='%.4f') # Usar la ruta completa
        print(f"Predicciones guardadas exitosamente en: {output_file_path}") # Mostrar ruta completa
        # Mostrar una vista previa
        print("\nVista previa de las predicciones:")
        print(results.head().to_string(index=False))
    except Exception as e:
        print(f"Error guardando el archivo de salida en {output_file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice clientes con alta probabilidad de compra para fechas dadas.")
    parser.add_argument(
        '--dates',
        required=True,
        nargs='+',
        help="Fecha(s) para predicción en formato YYYY-MM-DD (separadas por espacio)."
    )
    parser.add_argument(
        '--config',
        default='params.yaml',
        help='Ruta al archivo de configuración params.yaml (default: params.yaml)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Umbral de probabilidad para considerar una compra (ej: 0.5). Sobrescribe el valor en params.yaml.'
    )
    parser.add_argument(
        '--output',
        default='predictions.csv', # Nombre del archivo, no la ruta completa
        help='Nombre del archivo CSV de salida (se guardará en la carpeta predictions/, default: predictions.csv)'
    )

    args = parser.parse_args()
    # Pasar el nombre del archivo, la función construye la ruta completa
    predict_high_probability_clients(args.config, args.dates, args.threshold, args.output)