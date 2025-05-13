# src/add_cluster_recommendations.py
import pandas as pd
import argparse
import yaml
from pathlib import Path


def load_config(config_path="params.yaml"):
    """Carga la configuración desde un archivo YAML."""
    print(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Añade recomendaciones de clustering a un archivo de predicciones.")
    parser.add_argument("--input_predictions_file", required=True,
                        help="Ruta al archivo CSV de predicciones de compra.")
    parser.add_argument("--output_file", required=True,
                        help="Ruta para guardar el archivo CSV final con recomendaciones.")
    parser.add_argument("--config", default="params.yaml", help="Ruta al archivo de configuración YAML principal.")
    args = parser.parse_args()

    print(f"Leyendo configuracion desde: {args.config}")
    config = load_config(args.config)

    # Obtener rutas del config
    base_path = Path(config['data']['base_path'])
    processed_folder = config['data']['processed_folder']
    rec_cluster_cfg = config['recommendations_clustering']

    cluster_recs_folder = base_path / processed_folder / rec_cluster_cfg['output_folder']
    precomputed_cluster_recs_path = cluster_recs_folder / rec_cluster_cfg['precomputed_cluster_recs_file']

    try:
        print(f"Leyendo predicciones de compra desde: {args.input_predictions_file}")
        df_predictions = pd.read_csv(args.input_predictions_file)
        # Asumir que la columna de cliente en predicciones es 'client' o 'id_client'
        # y necesita ser string para el join
        if 'client' in df_predictions.columns:
            df_predictions['id_client_join'] = df_predictions['client'].astype(str)
        elif 'id_client' in df_predictions.columns:
            df_predictions['id_client_join'] = df_predictions['id_client'].astype(str)
        else:
            raise ValueError("La columna 'client' o 'id_client' no se encuentra en el archivo de predicciones.")

        print(f"Leyendo recomendaciones precalculadas por clustering desde: {precomputed_cluster_recs_path}")
        if not precomputed_cluster_recs_path.exists():
            print(f"Error: El archivo de recomendaciones por clustering '{precomputed_cluster_recs_path}' no existe.")
            print("Asegúrate de haber ejecutado primero el script de pre-cálculo de clustering.")
            return

        df_cluster_recs = pd.read_parquet(precomputed_cluster_recs_path)
        # Asegurar que la columna de cliente en las recomendaciones de clustering sea string
        df_cluster_recs['id_client_join'] = df_cluster_recs['id_client'].astype(str)

        # Unir las predicciones con las recomendaciones de clustering
        # Solo para los clientes que tienen predicción de compra Y recomendación de clustering
        print("Uniendo predicciones con recomendaciones de clustering...")
        df_final = pd.merge(
            df_predictions,
            df_cluster_recs[['id_client_join', 'purchase_pattern_cluster', 'recommended_products']],
            on='id_client_join',
            how='left'  # 'left' para mantener todas las predicciones y añadir recs si existen
            # 'inner' si solo quieres clientes con ambas
        )

        # Renombrar 'recommended_products' para evitar colisión si 'get_recommendations.py' usa el mismo nombre
        df_final.rename(columns={'recommended_products': 'cluster_recommended_products'}, inplace=True)
        df_final.drop(columns=['id_client_join'], inplace=True)  # Eliminar columna de join auxiliar

        # Guardar el archivo final
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Archivo final con predicciones y recomendaciones de clustering guardado en: {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado. {e}")
    except ValueError as e:
        print(f"Error de valor: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()