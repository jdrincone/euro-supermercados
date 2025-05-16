# src/train_recommender.py

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
from dateutil.relativedelta import relativedelta
import time


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def create_sparse_matrix(transactions_df):
    """Crea la matriz dispersa usuario-item y los mapeos."""
    print("Creando Matriz de Interacción Usuario-Item (basada en datos filtrados)...")
    valid_products = transactions_df['product'].unique()
    valid_clients = transactions_df['client'].unique()
    print(f"Clientes activos (post-filtros): {len(valid_clients)}")
    print(f"Productos activos (post-filtros): {len(valid_products)}")
    user_item_counts = transactions_df.groupby(['client', 'product']).size().reset_index(name='purchase_count')
    print(f"Interacciones únicas Usuario-Item (post-filtros): {len(user_item_counts)}")

    user_item_counts['client_id'] = user_item_counts['client'].astype('category').cat.codes
    user_item_counts['product_id'] = user_item_counts['product'].astype('category').cat.codes

    user_map = dict(enumerate(user_item_counts['client'].astype('category').cat.categories))
    product_map = dict(enumerate(user_item_counts['product'].astype('category').cat.categories))
    user_map_inv = {v: k for k, v in user_map.items()}
    product_map_inv = {v: k for k, v in product_map.items()}

    print(f"Usuarios únicos en matriz: {len(user_map)}, Productos únicos en matriz: {len(product_map)}")

    sparse_user_item = csr_matrix(
        (user_item_counts['purchase_count'],
         (user_item_counts['client_id'], user_item_counts['product_id'])),
        shape=(len(user_map), len(product_map))
    )

    if sparse_user_item.shape[0] > 0 and sparse_user_item.shape[1] > 0:
        sparsity = 1.0 - (sparse_user_item.nnz / (sparse_user_item.shape[0] * sparse_user_item.shape[1]))
        print(f"Matriz creada. Forma: {sparse_user_item.shape}, Densidad: {1.0 - sparsity:.4f}")
    else:
        print("Matriz creada. Forma: (0,0) - No hay datos suficientes tras el filtro.")
        sparse_user_item = None

    mappings = {
        'user_map': user_map, 'product_map': product_map,
        'user_map_inv': user_map_inv, 'product_map_inv': product_map_inv
    }
    return sparse_user_item, mappings

def calculate_item_similarity(sparse_user_item):
    """Calcula la matriz de similitud item-item."""
    if sparse_user_item is None or sparse_user_item.shape[1] < 2:
         print("\nMatriz usuario-item vacía o con < 2 productos, no se puede calcular similitud.")
         return None
    print("\nCalculando similitud Item-Item (Coseno)...")
    try:
        item_similarity = cosine_similarity(sparse_user_item.T, dense_output=False)
        print(f"Matriz de similitud calculada. Forma: {item_similarity.shape}")
        return item_similarity
    except Exception as e:
        print(f"Error calculando similitud: {e}")
        return None

def get_top_n_recommendations(client_idx, sparse_user_item_matrix, similarity_matrix, n=10):
    """Obtiene las N recomendaciones principales para un cliente."""
    if client_idx >= sparse_user_item_matrix.shape[0]:
        return []

    client_purchases = sparse_user_item_matrix[client_idx, :]
    purchased_items_indices = client_purchases.indices
    purchase_values = client_purchases.data

    if len(purchased_items_indices) == 0:
        return []

    scores = {}
    for i, item_index in enumerate(purchased_items_indices):
        if item_index >= similarity_matrix.shape[0]:
            continue
        similarity_row = similarity_matrix[item_index].toarray().flatten()
        for j, similar_item_score in enumerate(similarity_row):
            if j != item_index and j not in purchased_items_indices and similar_item_score > 0:
                scores[j] = scores.get(j, 0) + similar_item_score * purchase_values[i]

    ranked_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_index for item_index, _ in ranked_recommendations[:n]]

def train_and_save(config_path):
    """Carga datos, filtra productos y fecha, entrena CF Item-Item, precalcula y guarda recomendaciones como lista de nombres de productos."""
    start_time = time.time()
    print("--- Iniciando Entrenamiento y Pre-cálculo del Recomendador Item-Item ---")
    print("--- Filtros aplicados: Top 5 productos por cliente (último mes) ---")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Rutas ---
    rec_params = config['recommendations_item_item']
    data_params = config['data']
    transaction_file = Path(data_params['base_path']) / data_params['processed_folder'] / rec_params['transaction_data_processed_file']
    products_file_path = Path(data_params['base_path']) / data_params['raw_folder'] / data_params['products_file'] # Ruta al archivo de productos
    model_dir = Path(rec_params['model_folder'])
    output_dir = Path(rec_params['output_folder'])
    similarity_file = model_dir / rec_params['similarity_matrix_file']
    mappings_file = model_dir / rec_params['mappings_file']
    user_item_file = model_dir / rec_params['user_item_matrix_file']
    precomputed_recs_file = output_dir / rec_params['precomputed_recs_file']
    num_recommendations = rec_params['num_recommendations']

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar Datos ---
    print(f"\nCargando TODAS las transacciones desde: {transaction_file}")
    transactions_df_full = pd.read_parquet(transaction_file)
    transactions_df_full.rename(columns={'id_client': 'client',
                                         'product': 'product',
                                         'date_sale': 'date'}, inplace=True)

    transactions_df_full.dropna(subset=['client', 'product', 'date'], inplace=True)
    transactions_df_full['client'] = transactions_df_full['client'].astype(str)
    transactions_df_full['product'] = transactions_df_full['product'].astype(str).str.strip()
    print(f"Cargadas {len(transactions_df_full)} filas totales.")

    # --- FILTRADO POR FECHA INICIAL ---
    max_date = transactions_df_full['date'].max()
    last_month = data_params.get('last_month_with_sale', 3)
    cutoff_date = max_date - relativedelta(months=last_month)
    transactions_last_month = transactions_df_full[transactions_df_full['date'] >= cutoff_date].copy()
    print(f"Fecha máxima en datos: {max_date.strftime('%Y-%m-%d')}")
    print(f"Fecha de corte ({last_month} meses): {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Filas después de filtrar por fecha: {len(transactions_last_month)}")

    # --- FILTRADO DE TOP 5 PRODUCTOS POR CLIENTE ---
    print("\nSeleccionando el top 5 de productos comprados por cada cliente en el último mes...")
    if not transactions_last_month.empty:
        client_product_counts = transactions_last_month.groupby(['client', 'product']).size().reset_index(name='purchase_count')
        client_product_counts_sorted = client_product_counts.sort_values(['client', 'purchase_count'], ascending=[True, False])
        top_5_products_per_client = client_product_counts_sorted.groupby('client').head(5).reset_index(drop=True)
        print(f"Número de interacciones cliente-producto después del filtro de top 5: {len(top_5_products_per_client)}")

        transactions_df_filtered = pd.merge(transactions_last_month[['client', 'product', 'date']],
                                               top_5_products_per_client[['client', 'product']],
                                               on=['client', 'product'],
                                               how='inner').drop_duplicates().copy()
        print(f"Filas en el DataFrame filtrado final para la matriz: {len(transactions_df_filtered)}")

        if transactions_df_filtered.empty:
            print("Advertencia: El DataFrame filtrado para la matriz está vacío.")
            return
    else:
        print("No hay transacciones en el último mes. No se puede continuar con el filtrado de productos.")
        return

    # ----------------------------------

    # --- Crear Matriz User-Item (con datos filtrados) ---
    sparse_user_item, mappings = create_sparse_matrix(transactions_df_filtered)
    if sparse_user_item is None:
        print("Matriz Usuario-Item no pudo ser creada con datos filtrados.")
        return

    # --- Calcular Similitud Item-Item ---
    item_similarity = calculate_item_similarity(sparse_user_item)
    if item_similarity is None:
        print("Matriz de Similitud no pudo ser calculada.")
        return

    # --- Cargar Datos de Productos para Mapeo ID a Nombre ---
    print(f"\nCargando datos de productos desde: {products_file_path}")
    try:
        productos_df = pd.read_csv(
            products_file_path,
            dtype={"codigo_unico": str}
        ).rename(columns={"codigo_unico": "product", "description": "product_name"})
        productos_df = productos_df.loc[:, ["product", "product_name"]].drop_duplicates("product")
        productos_df['product'] = productos_df['product'].str.strip()
        product_name_map = productos_df.set_index('product')['product_name'].to_dict()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de productos en: {products_file_path}. Las recomendaciones contendrán IDs en lugar de nombres.")
        product_name_map = {}

    # --- Generar Recomendaciones y Formatear para Guardar ---
    print(f"\nGenerando {num_recommendations} recomendaciones para los {len(mappings['user_map'])} clientes activos...")
    recommendation_list = []
    for client_str_id, client_idx in mappings['user_map_inv'].items():
        recommended_item_indices = get_top_n_recommendations(
            client_idx, sparse_user_item, item_similarity, n=num_recommendations
        )
        recommended_products_names = []
        for item_index in recommended_item_indices:
            product_id = mappings['product_map'].get(item_index)
            if product_id:
                product_name = product_name_map.get(product_id, product_id) # Usa ID si no encuentra el nombre
                recommended_products_names.append(product_name)

        recommendation_list.append({
            'client': client_str_id,
            'recommended_products': recommended_products_names
        })

    recommendations_df_to_save = pd.DataFrame(recommendation_list)

    # --- Guardar Artefactos ---
    print("\nGuardando artefactos del modelo y recomendaciones pre-calculadas...")
    try:
        if item_similarity is not None:
             save_npz(similarity_file, item_similarity)
             print(f"  - Matriz de similitud guardada en: {similarity_file}")
        else: print("  - No se guardó matriz de similitud (no calculada).")

        if mappings:
             with open(mappings_file, 'wb') as f:
                 pickle.dump(mappings, f)
             print(f"  - Mapeos guardados en: {mappings_file}")
        else: print("  - No se guardaron mapeos.")

        if sparse_user_item is not None:
             save_npz(user_item_file, sparse_user_item)
             print(f"  - Matriz User-Item (filtrada) guardada en: {user_item_file}")
        else: print("  - No se guardó matriz User-Item.")

        if not recommendations_df_to_save.empty:
            recommendations_df_to_save.to_parquet(precomputed_recs_file, index=False)
            print(f"  - Recomendaciones pre-calculadas (lista de nombres) guardadas en: {precomputed_recs_file}")
        else:
            print("  - No se guardó archivo de recomendaciones pre-calculadas (estaba vacío).")

        end_time = time.time()
        training_time = end_time - start_time
        print(f"\n--- Proceso de Entrenamiento y Pre-cálculo (Filtro de top 5) Finalizado en {training_time:.2f} segundos ---")

    except Exception as e:
        print(f"Error guardando los artefactos: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena recomendador Item-Item CF (filtrado por top 5) y pre-calcula recomendaciones.")
    parser.add_argument(
        '--config',
        default='params.yaml',
        help='Ruta al archivo de configuración params.yaml'
    )
    args = parser.parse_args()
    train_and_save(args.config)