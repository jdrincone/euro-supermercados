# src/add_recommendations.py
import pandas as pd
import argparse
import logging
from pathlib import Path

# Configuración básica del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Añade recomendaciones de clustering e item-item a un archivo de predicciones de compra.
    """
    parser = argparse.ArgumentParser(
        description="Añade recomendaciones de clustering e item-item a un archivo de predicciones."
    )
    parser.add_argument(
        "--input_predictions_file",
        required=True,
        help="Nombre del archivo CSV de predicciones de compra (ej: predictions.csv).",
    )
    parser.add_argument(
        "--output_cluster_file",
        required=True,
        help="Nombre del archivo CSV final con recomendaciones de clustering (ej: predictions_with_rec_cluster.csv).",
    )
    parser.add_argument(
        "--output_item_item_file",
        required=True,
        help="Nombre del archivo CSV final con recomendaciones item-item (ej: predictions_with_rec_item_item.csv).",
    )
    args = parser.parse_args()

    # Definición de rutas directamente
    predictions_input_path = Path('predictions') / args.input_predictions_file
    precomputed_cluster_recs_path = Path('recommendations') / 'precomputed_cluster_recs.parquet'
    precomputed_item_item_recs_path = Path('recommendations') / 'precomputed_item_item_recs.parquet'
    output_filepath_recs_cluster = Path('predictions') / args.output_cluster_file
    output_filepath_recs_item_item = Path('predictions') / args.output_item_item_file

    try:
        logging.info(f"Intentando leer predicciones de compra desde: {predictions_input_path}")
        df_predictions = pd.read_csv(predictions_input_path)

        # Identificar la columna de cliente
        client_column_predictions = None
        if 'client' in df_predictions.columns:
            client_column_predictions = 'client'
        elif 'id_client' in df_predictions.columns:
            client_column_predictions = 'id_client'

        if not client_column_predictions:
            raise ValueError("No se encontró la columna 'client' ni 'id_client' en el archivo de predicciones.")

        df_predictions['client_join'] = df_predictions[client_column_predictions].astype(str)

        # --- Procesamiento de recomendaciones de CLUSTERING ---
        logging.info(f"Leyendo recomendaciones precalculadas por clustering desde: {precomputed_cluster_recs_path}")
        if precomputed_cluster_recs_path.exists():
            df_cluster_recs = pd.read_parquet(precomputed_cluster_recs_path)
            client_column_cluster = None
            if 'client_id' in df_cluster_recs.columns:
                client_column_cluster = 'client_id'
            elif 'client' in df_cluster_recs.columns:
                client_column_cluster = 'client'

            if client_column_cluster:
                df_cluster_recs['client_join'] = df_cluster_recs[client_column_cluster].astype(str)

                logging.info("Uniendo predicciones con recomendaciones de clustering...")
                df_final_cluster = pd.merge(
                    df_predictions.copy(),
                    df_cluster_recs[['client_join', 'recommended_products']],
                    on='client_join',
                    how='left'
                )
                df_final_cluster.rename(columns={'recommended_products': 'cluster_recommended_products'}, inplace=True)
                df_final_cluster.drop(columns=['client_join'], inplace=True)

                # Guardar el archivo final con recomendaciones de clustering
                output_filepath_recs_cluster.parent.mkdir(parents=True, exist_ok=True)
                df_final_cluster.to_csv(output_filepath_recs_cluster, index=False)
                logging.info(
                    f"Archivo con predicciones y recomendaciones de clustering guardado en: {output_filepath_recs_cluster}"
                )
            else:
                logging.warning(f"No se encontró la columna 'client_id' ni 'client' en el archivo de recomendaciones de clustering: {precomputed_cluster_recs_path}. No se pudieron añadir estas recomendaciones.")
        else:
            logging.warning(f"No se encontró el archivo de recomendaciones de clustering: {precomputed_cluster_recs_path}. No se añadirán estas recomendaciones.")

        # --- Procesamiento de recomendaciones ITEM-ITEM ---
        logging.info(f"Leyendo recomendaciones precalculadas item-item desde: {precomputed_item_item_recs_path}")
        if precomputed_item_item_recs_path.exists():
            df_item_item_recs = pd.read_parquet(precomputed_item_item_recs_path)
            client_column_item_item = None
            if 'client_id' in df_item_item_recs.columns:
                client_column_item_item = 'client_id'
            elif 'client' in df_item_item_recs.columns:
                client_column_item_item = 'client'

            if client_column_item_item:
                df_item_item_recs['client_join'] = df_item_item_recs[client_column_item_item].astype(str)

                logging.info("Uniendo predicciones con recomendaciones item-item...")
                df_final_item_item = pd.merge(
                    df_predictions.copy(),
                    df_item_item_recs[['client_join', 'recommended_products']],
                    on='client_join',
                    how='left'
                )
                df_final_item_item.rename(columns={'recommended_products': 'item_item_recommended_products'}, inplace=True)
                df_final_item_item.drop(columns=['client_join'], inplace=True)

                # Guardar el archivo final con recomendaciones item-item
                output_filepath_recs_item_item.parent.mkdir(parents=True, exist_ok=True)
                df_final_item_item.to_csv(output_filepath_recs_item_item, index=False)
                logging.info(
                    f"Archivo con predicciones y recomendaciones item-item guardado en: {output_filepath_recs_item_item}"
                )
            else:
                logging.warning(f"No se encontró la columna 'client_id' ni 'client' en el archivo de recomendaciones item-item: {precomputed_item_item_recs_path}. No se pudieron añadir estas recomendaciones.")
        else:
            logging.warning(f"No se encontró el archivo de recomendaciones item-item: {precomputed_item_item_recs_path}. No se añadirán estas recomendaciones.")

    except FileNotFoundError as e:
        logging.critical(f"Error CRÍTICO: Archivo no encontrado: {e.filename}")
        logging.critical("Verifica que los archivos de predicciones y recomendaciones existan en las rutas especificadas.")
    except ValueError as e:
        logging.error(f"Error de valor: {e}")
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()