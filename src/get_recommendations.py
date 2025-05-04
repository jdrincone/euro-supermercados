# src/get_recommendations.py

import pandas as pd
import yaml
from pathlib import Path
import argparse
import sys # Para salir si hay error

def load_and_get_recs(clients_input_df, config_path, output_file):
    """
    Carga recomendaciones pre-calculadas y las filtra/une para los clientes dados.
    """
    print("--- Iniciando Búsqueda de Recomendaciones Pre-calculadas ---")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Rutas ---
    rec_params = config['recommendations']
    data_params = config['data']
    precomputed_recs_file = Path(rec_params['output_folder']) / rec_params['precomputed_recs_file']
    products_file = Path(data_params['base_path']) / data_params['raw_folder'] / data_params['products_file']

    # --- Cargar Recomendaciones Pre-calculadas ---
    try:
        print(f"Cargando recomendaciones desde: {precomputed_recs_file}")
        all_recs_df = pd.read_parquet(precomputed_recs_file)
        all_recs_df['client'] = all_recs_df['client'].astype(str) # Asegurar tipo
        print(f"Cargadas {len(all_recs_df)} filas de recs para {all_recs_df['client'].nunique()} clientes.")
    except FileNotFoundError:
        print(f"Error Fatal: Archivo de recomendaciones pre-calculadas no encontrado en {precomputed_recs_file}")
        print("Por favor, ejecuta 'python src/train_recommender.py' primero.")
        sys.exit(1) # Salir del script
    except Exception as e:
        print(f"Error Fatal: No se pudo cargar el archivo de recomendaciones: {e}")
        sys.exit(1)

    # --- Preparar DataFrame de Entrada ---
    if 'client' not in clients_input_df.columns:
        print("Error Fatal: El archivo de entrada debe contener la columna 'client'.")
        sys.exit(1)
    clients_input_df['client'] = clients_input_df['client'].astype(str) # Asegurar tipo

    # --- Buscar Recomendaciones (Merge) ---
    print(f"Buscando recomendaciones para {len(clients_input_df)} filas de entrada (representando {clients_input_df['client'].nunique()} clientes únicos)...")
    # Unir las recomendaciones pre-calculadas con el DataFrame de entrada
    results_df = pd.merge(
        clients_input_df,
        all_recs_df,
        on='client',
        how='left' # Mantener todas las filas de entrada, añadir recs si existen
    )

    found_clients = results_df[results_df['recommendation_rank'].notna()]['client'].nunique()
    print(f"Se encontraron recomendaciones para {found_clients} clientes únicos de la entrada.")

    # --- Opcional: Añadir Detalles del Producto ---
    try:
        print(f"Cargando detalles del producto desde: {products_file}")
        productos = pd.read_csv(products_file, dtype={"codigo_unico": str})
        productos.rename(columns={"codigo_unico": "product"}, inplace=True)
        productos['product'] = productos['product'].str.strip()
        product_cols = ['product', 'description', 'brand', 'category']
        productos = productos[product_cols].drop_duplicates(subset=['product'])
        # Renombrar para unir con la columna de recomendaciones
        productos.rename(columns={'product': 'recommended_product'}, inplace=True)

        # Eliminar columnas de detalles si ya existen antes del merge
        cols_to_drop = ['description', 'brand', 'category']
        cols_exist = [col for col in cols_to_drop if col in results_df.columns]
        if cols_exist: results_df = results_df.drop(columns=cols_exist)

        # Unir detalles
        results_df = pd.merge(
            results_df,
            productos,
            on='recommended_product',
            how='left' # Mantener todas las filas, incluso si no hay detalles del producto
        )
        print("Detalles del producto añadidos.")
    except FileNotFoundError:
        print(f"Advertencia: Archivo de productos no encontrado en {products_file}. No se añadirán detalles.")
    except Exception as e:
        print(f"Advertencia: No se pudieron añadir detalles del producto: {e}")

    # --- Guardar Resultados ---
    try:
        # Ordenar para mejor visualización
        results_df = results_df.sort_values(
            # Ordenar por columnas originales + ranking
            list(clients_input_df.columns) + ['recommendation_rank'],
            na_position='last' # Poner filas sin recs al final
        )
        output_path = Path(output_file)
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nResultados con recomendaciones guardados exitosamente en: {output_path}")
        print("\nVista previa de la salida:")
        print(results_df.head(15).to_string()) # Mostrar más filas para ver estructura
    except Exception as e:
        print(f"Error guardando el archivo de salida en {output_path}: {e}")

    print("\n--- Búsqueda de Recomendaciones Finalizada ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca recomendaciones pre-calculadas para una lista de clientes.")
    parser.add_argument(
        '--input_file',
        required=True,
        help='Ruta al archivo CSV de entrada con la columna "client" (ej: salida de predict.py).'
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Ruta al archivo CSV de salida donde se guardarán los resultados con recomendaciones.'
    )
    parser.add_argument(
        '--config',
        default='params.yaml',
        help='Ruta al archivo de configuración params.yaml.'
    )
    args = parser.parse_args()

    try:
        input_clients_df = pd.read_csv(args.input_file)
        load_and_get_recs(input_clients_df, args.config, args.output_file)
    except FileNotFoundError:
        print(f"Error Fatal: Archivo de entrada no encontrado en {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error Fatal procesando el archivo de entrada: {e}")
        sys.exit(1)