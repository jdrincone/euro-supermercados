import pandas as pd
import yaml
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import joblib
import os
from dotenv import load_dotenv

from utils import obtener_token, obtener_terceros

load_dotenv()

def load_config(config_path: str) -> dict:
    """Loads the configuration from the specified YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_features(config: dict) -> tuple:
    """Loads the calibrated model and feature calendar."""
    data_params = config['data']
    model_params = config['model']
    processed_path = Path(data_params['base_path']) / data_params['processed_folder']
    model_path = Path(model_params['model_dir'])
    features_file = processed_path / config['featurize']['output_file']
    calibrated_model_file = model_path / model_params['calibrated_model_name']

    try:
        print(f"Cargando modelo calibrado desde: {calibrated_model_file}")
        calibrator = joblib.load(calibrated_model_file)
        print(f"Cargando calendario de features desde: {features_file}")
        calendar = pd.read_parquet(features_file)
        calendar['date'] = pd.to_datetime(calendar['date'])
        calendar['client'] = calendar['client'].astype(str)
        return calibrator, calendar
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}. Asegúrate de haber ejecutado `dvc repro`. Detalle: {e}")
        raise
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        raise

def process_prediction_dates(prediction_dates_str: list) -> list:
    """Converts the date strings to datetime objects."""
    try:
        prediction_dates = [pd.to_datetime(d).normalize() for d in prediction_dates_str]
        print(f"Fechas para predicción: {[d.strftime('%Y-%m-%d') for d in prediction_dates]}")
        return prediction_dates
    except ValueError as e:
        print(f"Error: Formato de fecha inválido. Use<ctrl98>-MM-DD. Detalle: {e}")
        raise

def make_predictions(config: dict, calendar: pd.DataFrame, prediction_dates: list, threshold: float) -> pd.DataFrame:
    """Makes predictions for the specified dates."""
    train_params = config['train']
    predict_df = calendar[calendar['date'].isin(prediction_dates)].copy()

    if predict_df.empty:
        print("Error: No se encontraron datos para las fechas especificadas.")
        print(f"Rango de fechas en calendario: {calendar['date'].min().date()} a {calendar['date'].max().date()}")
        return pd.DataFrame()

    print(f"Realizando predicciones para {len(predict_df)} registros cliente-día...")
    features = train_params['features']
    try:
        calibrator, _ = load_model_and_features(config) # Reload to ensure it's in scope
        predict_df['prob'] = calibrator.predict_proba(predict_df[features])[:, 1]
        high_prob_clients_df = predict_df[predict_df['prob'] >= threshold].copy()
        return high_prob_clients_df
    except ValueError as e:
        print(f"Error durante la predicción. ¿Faltan features o tipos incorrectos? Detalle: {e}")
        print("Features esperadas:", features)
        print("Features encontradas:", predict_df[features].columns.tolist())
        print("Tipos de datos encontrados:\n", predict_df[features].dtypes)
        raise
    except Exception as e:
        print(f"Error inesperado durante la predicción: {e}")
        raise

def get_customer_info(predicted_clients: pd.Series) -> pd.DataFrame:
    """Retrieves customer contact information using the API."""
    username = os.environ.get('API_USERNAME')
    password = os.environ.get('API_PASSWORD')
    token = obtener_token(username, password)
    terceros = obtener_terceros(token, predicted_clients, batch_size=10)
    df_terceros = pd.DataFrame(terceros)
    df_terceros.rename(columns={'document': 'client'}, inplace=True)
    return df_terceros

def load_sales_and_products_data(config: dict) -> pd.DataFrame:
    """Loads sales and products data from CSV files."""
    data_params = config['data']
    raw_path = Path(data_params['base_path']) / data_params['raw_folder']

    try:
        df_ventas = pd.read_csv(
            raw_path / data_params['sales_file'],
            usecols=['identification_doct', 'product', 'date_sale'])
        df_ventas['client'] = df_ventas['identification_doct'].astype(str).str.strip()
        mask_digits_ventas = df_ventas["client"].str.isdigit().fillna(False)
        mask_zero_ventas = ~df_ventas["client"].str.startswith("0", na=False)
        df_ventas = df_ventas[mask_digits_ventas & mask_zero_ventas].copy()
        df_ventas['product'] = df_ventas['product'].astype(str).str.strip()
        df_ventas['date_sale'] = pd.to_datetime(df_ventas['date_sale']).dt.normalize()
        df_ventas = df_ventas.dropna(subset=['date_sale'])

        productos_df = pd.read_csv(
            raw_path / data_params['products_file'],
            dtype={"codigo_unico": str}
        ).rename(columns={"codigo_unico": "product"})
        productos_df = productos_df.loc[:, ["product", "description"]]
        productos_df = productos_df.drop_duplicates("product")
        productos_df['product'] = productos_df['product'].str.strip()

        ventas_prod = pd.merge(df_ventas, productos_df, on="product")

        return ventas_prod
    except FileNotFoundError as e:
        print(f"Error al cargar archivos de ventas o productos: {e}")
        raise
    except Exception as e:
        print(f"Error inesperado al cargar datos: {e}")
        raise

def filter_excluded_products(df_ventas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filters out excluded product descriptions from the sales data."""
    clustering_params = config['recommendations_clustering']
    excluded_products = clustering_params.get('excluded_product_descriptions', [])
    df_ventas = df_ventas[~df_ventas['description'].isin(excluded_products)]
    return df_ventas

def get_top_n_percent_products(group: pd.DataFrame, percentile: float = 0.25) -> pd.DataFrame:
    """Gets the top percentile of products based on relative frequency."""
    n = max(1, int(len(group) * percentile))  # al menos 1
    return group.head(n)

def generate_product_recommendations(
        df_ventas: pd.DataFrame,
        predicted_clients: pd.Series,
        recommendation_months: int = 1,
        top_percentile: float = 0.25) -> pd.DataFrame:
    """Generates product recommendations based on recent purchase history."""
    last_months_ago = datetime.now().date() - timedelta(days=recommendation_months * 30)
    recent_purchases = df_ventas[df_ventas["date_sale"].dt.date >= last_months_ago]
    recent_purchases_predicted = recent_purchases[recent_purchases["client"].isin(predicted_clients)]

    if recent_purchases_predicted.empty:
        print("No hay compras recientes para los clientes predichos.")
        return pd.DataFrame({'client': predicted_clients, 'recommended_products': 'Sin recomendaciones recientes'})

    # Calcular frecuencia relativa de productos comprados por cliente
    freq = (
        recent_purchases_predicted.groupby(['client', 'description'])
        .size()
        .groupby(level=0)
        .transform(lambda x: x / x.sum())
        .reset_index(name='rel_freq')
    )

    # Ordenar por cliente y frecuencia relativa descendente
    freq_sorted = freq.sort_values(['client', 'rel_freq'], ascending=[True, False])

    # Obtener el top percentile de productos más frecuentes por cliente
    top_products = freq_sorted.groupby('client', group_keys=False).apply(
        get_top_n_percent_products,
        percentile=top_percentile)

    # Summarizar los productos recomendados por cliente
    recommendations = (
        top_products
        .groupby('client')
        .agg(recommended_products=('description', lambda x: ', '.join(x)),
             count_products=('description', 'count'))
        .reset_index()
    )
    return recommendations

def merge_predictions_and_recommendations(predictions_df: pd.DataFrame, customer_info_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the prediction results with customer info and product recommendations."""
    predictions_with_contact = pd.merge(
        predictions_df.loc[:, ['date', 'client', "prob"]],
        customer_info_df.loc[:, ['name', 'email', 'phone', 'telephone', 'client']],
        on='client')
    final_predictions = pd.merge(predictions_with_contact, recommendations_df, on="client", how="left")
    final_predictions['recommended_products'].fillna('Sin recomendaciones recientes', inplace=True)
    return final_predictions

def save_predictions(final_predictions_df: pd.DataFrame, output_file_path: Path):
    """Saves the final predictions with recommendations to a CSV file."""
    print(f"Guardando predicciones con recomendaciones en: {output_file_path}")
    final_predictions_df['date'] = final_predictions_df['date'].dt.strftime('%Y-%m-%d')
    final_predictions_df = final_predictions_df.sort_values(['date', 'prob'], ascending=[True, False])
    try:
        final_predictions_df.to_csv(output_file_path, index=False, float_format='%.4f')
        print("Predicciones guardadas exitosamente.")
        print("\nVista previa de las predicciones:")
        print(final_predictions_df.head().to_string(index=False))
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}")
        raise

def main(
        config_path: str,
        prediction_dates_str: list,
        threshold_override: float = None,
        output_filename: str = 'predictions_with_recommendations.csv',
        recommendation_months: int = 1):
    """Main function to orchestrate the prediction and recommendation process."""
    try:
        config = load_config(config_path)
        calibrator, calendar = load_model_and_features(config)
        prediction_dates = process_prediction_dates(prediction_dates_str)

        eval_params = config['evaluate']
        threshold = threshold_override if threshold_override is not None else eval_params['evaluation_threshold']
        print(f"Usando umbral de probabilidad: {threshold:.2f}")

        predicted_clients_df = make_predictions(config, calendar, prediction_dates, threshold)
        if predicted_clients_df.empty:
            print("No se encontraron clientes con alta probabilidad de compra.")
            return

        predicted_clients = predicted_clients_df["client"].unique()
        customer_info_df = get_customer_info(predicted_clients)

        df_ventas = load_sales_and_products_data(config)
        print(df_ventas.head())
        df_ventas = filter_excluded_products(df_ventas, config)
        print("Filtro")

        recommendation_params = config.get('recommendation_settings', {})
        top_percentile = recommendation_params.get('top_product_percentile', 0.25)

        recommendations_df = generate_product_recommendations(
            df_ventas,
            predicted_clients,
            recommendation_months,
            top_percentile)
        print("recommendations_df", recommendations_df)

        final_predictions_df = merge_predictions_and_recommendations(
            predicted_clients_df,
            customer_info_df,
            recommendations_df)

        predictions_dir = Path('predictions')
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = predictions_dir / output_filename
        save_predictions(final_predictions_df, output_file_path)

    except FileNotFoundError:
        print("Error: Archivo de configuración o datos no encontrado.")
    except Exception as e:
        print(f"Ocurrió un error durante la ejecución: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predice clientes con alta probabilidad de compra y genera recomendaciones.")
    parser.add_argument(
        '--dates',
        required=True,
        nargs='+',
        help="Fecha(s) para predicción en formato<ctrl98>-MM-DD (separadas por espacio)."
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
        default='predictions_with_recommendations.csv',
        help='Nombre del archivo CSV de salida (se guardará en la carpeta predictions/, default: predictions_with_recommendations.csv)'
    )
    parser.add_argument(
        '--recommendation_months',
        type=int,
        default=1,
        help='Número de meses hacia atrás para considerar compras recientes para recomendaciones (default: 1).'
    )

    args = parser.parse_args()
    main(args.config, args.dates, args.threshold, args.output, args.recommendation_months)