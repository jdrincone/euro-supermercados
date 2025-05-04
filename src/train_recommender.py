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

def get_recs_for_client(client_idx, client_str_id, sparse_user_item_matrix, similarity_matrix,
                         product_idx_to_str, n_recs):
    """Genera recomendaciones para un solo cliente."""
    recommendations = {} # {product_idx: score}
    if client_idx >= sparse_user_item_matrix.shape[0]: return []

    client_purchases_sparse = sparse_user_item_matrix[client_idx, :]
    purchased_product_indices = client_purchases_sparse.indices
    purchase_values = client_purchases_sparse.data

    if len(purchased_product_indices) == 0: return []

    # Pre-filtrar índices de productos comprados que sean válidos en la matriz de similitud
    valid_purchased_indices = [idx for idx in purchased_product_indices if idx < similarity_matrix.shape[0]]
    if not valid_purchased_indices: return []

    # Obtener las filas de similitud relevantes de una vez (puede ser más eficiente)
    relevant_sim_rows = similarity_matrix[valid_purchased_indices, :]

    # Iterar sobre los índices válidos y sus valores correspondientes
    for i, purchased_idx in enumerate(valid_purchased_indices):
        # Encontrar el valor de compra correspondiente (puede requerir buscar el índice original)
        original_index_pos = np.where(purchased_product_indices == purchased_idx)[0][0]
        purchase_count = purchase_values[original_index_pos]

        # Obtener la fila de similitud ya extraída
        sim_row = relevant_sim_rows[i, :].toarray().flatten()

        for similar_item_idx, similarity_score in enumerate(sim_row):
            if similar_item_idx >= similarity_matrix.shape[1]: continue # Verificar índice

            if similar_item_idx != purchased_idx and similarity_score > 0:
                 # Usar el set de índices válidos para la comprobación
                if similar_item_idx not in valid_purchased_indices:
                    current_score = recommendations.get(similar_item_idx, 0)
                    recommendations[similar_item_idx] = current_score + (similarity_score * purchase_count)

    sorted_recs_idx = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)

    final_recs_list = []
    for product_idx, score in sorted_recs_idx:
         product_str = product_idx_to_str.get(product_idx)
         if product_str:
             final_recs_list.append({
                 'client': client_str_id,
                 'recommended_product': product_str,
                 'recommendation_score': score
             })
         if len(final_recs_list) >= n_recs: break
    return final_recs_list


def train_and_save(config_path):
    """Carga datos, filtra productos y fecha, entrena CF Item-Item, precalcula y guarda."""
    print("--- Iniciando Entrenamiento y Pre-cálculo del Recomendador Item-Item ---")
    print("--- Filtros aplicados: Productos específicos + Últimos 3 Meses ---") # Mensaje actualizado
    start_time = time.time()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Rutas ---
    rec_params = config['recommendations']
    data_params = config['data']
    transaction_file = Path(data_params['base_path']) / data_params['processed_folder'] / rec_params['transaction_data_processed_file']
    products_file = Path(data_params['base_path']) / data_params['raw_folder'] / data_params['products_file']
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
    transactions_df_full.rename(columns={'id_client': 'client', # Asume que ya se llaman así
                                         'product': 'product',
                                         'date_sale': 'date'}, inplace=True)

    transactions_df_full.dropna(subset=['client', 'product', 'date'], inplace=True)
    transactions_df_full['client'] = transactions_df_full['client'].astype(str)
    transactions_df_full['product'] = transactions_df_full['product'].astype(str).str.strip()
    print(f"Cargadas {len(transactions_df_full)} filas totales.")

    print(f"\nCargando catálogo de productos desde: {products_file}")
    productos = pd.read_csv(
        products_file,
        usecols=["codigo_unico", "description"],
        dtype={"codigo_unico": str})
    productos.rename(columns={"codigo_unico": "product"}, inplace=True)
    productos['product'] = productos['product'].str.strip()

    productos = productos.drop_duplicates(subset=['product']).copy() # Usar .copy()
    print(f"Cargados {len(productos)} productos únicos en catálogo.")

    # --- INICIO: FILTRAR PRODUCTOS ---
    print("\nFiltrando productos a considerar para recomendación...")
    transactions_with_desc = pd.merge(
        transactions_df_full,
        productos,
        on='product',
        how='left'
    )

    # 2. Excluir productos específicos por descripción
    excluir_prd = ["BOLSA PLASTICA", "TRANSPORTE DOMICILIO"]
    print(f"Excluyendo descripciones: {excluir_prd}")
    transactions_filtered = transactions_with_desc[
        ~transactions_with_desc["description"].isin(excluir_prd) &
        transactions_with_desc["description"].notna()
    ].copy()
    print(f"Filas después de excluir por descripción: {len(transactions_filtered)}")

    if not transactions_filtered.empty:
        print("Calculando frecuencia por descripción de producto...")
        prod_desc_freq = transactions_filtered['description'].value_counts().reset_index(name='count')
        prod_desc_freq.rename(columns={'index': 'description'}, inplace=True) # Renombrar columna índice

        q50 = prod_desc_freq["count"].quantile(0.50)
        q100 = prod_desc_freq["count"].quantile(.90) #0.90 38seg--437 0.99
        print(f"Frecuencia por Descripción - Mediana (Q50): {q50}, Máxima (Q100): {q100}")

        # 5. Identificar descripciones dentro del rango Q50-Q100
        cond = prod_desc_freq["count"].between(q50, q100, inclusive="both")
        target_descriptions = prod_desc_freq[cond]["description"].unique()
        print(f"Número de descripciones seleccionadas (frecuencia entre Q50 y Q100): {len(target_descriptions)}")

        # 6. Obtener los IDs de producto (`product`) que corresponden a esas descripciones
        target_product_ids = productos[productos['description'].isin(target_descriptions)]['product'].unique()
        print(f"Número de IDs de producto únicos seleccionados: {len(target_product_ids)}")

        # 7. Filtrar el DataFrame de transacciones original (ya sin bolsas/transporte)
        # para quedarse solo con los productos seleccionados
        transactions_df_filtered_by_product = transactions_filtered[
            transactions_filtered['product'].isin(target_product_ids)
        ].copy() # Usar .copy()
        print(f"Filas después de filtrar por IDs de producto seleccionados: {len(transactions_df_filtered_by_product)}")
    else:
        print("No quedan transacciones tras excluir productos iniciales. No se puede continuar.")
        return


    max_date = transactions_df_filtered_by_product['date'].max()
    cutoff_date = max_date - relativedelta(months=1) # TODO pasar a parámeto 1, 2, 3 ?

    # DataFrame final a usar para la matriz
    transactions_df_final_filtered = transactions_df_filtered_by_product[
        transactions_df_filtered_by_product['date'] >= cutoff_date
    ].copy() # <--- DataFrame FILTRADO FINAL
    print(f"Fecha máxima en datos filtrados: {max_date.strftime('%Y-%m-%d')}")
    print(f"Fecha de corte (1 meses antes): {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Filas después de filtrar por fecha (y producto): {len(transactions_df_final_filtered)}")

    # ----------------------------------

    # --- Crear Matriz User-Item (con datos doblemente filtrados) ---
    # Pasamos el DF filtrado final
    sparse_user_item, mappings = create_sparse_matrix(transactions_df_final_filtered)
    if sparse_user_item is None:
        print("Matriz Usuario-Item no pudo ser creada con datos filtrados.")
        return

    # --- Calcular Similitud Item-Item ---
    item_similarity = calculate_item_similarity(sparse_user_item)
    if item_similarity is None:
        print("Matriz de Similitud no pudo ser calculada.")
        return

    # --- Generar Recomendaciones para TODOS los clientes (activos en los últimos 3 meses y con productos filtrados) ---
    print(f"\nGenerando {num_recommendations} recomendaciones para los {len(mappings['user_map_inv'])} clientes activos...")
    all_recommendations_list = []
    total_clients = len(mappings['user_map_inv'])
    processed_clients = 0

    # Iterar usando los mapeos generados a partir de los datos filtrados
    for client_str_id, client_idx in mappings['user_map_inv'].items():
        recs = get_recs_for_client(
            client_idx, client_str_id, sparse_user_item, item_similarity,
            mappings['product_map'], num_recommendations
        )
        all_recommendations_list.extend(recs)

        processed_clients += 1
        if processed_clients % 500 == 0 or processed_clients == total_clients:
             print(f"  Recomendaciones generadas para {processed_clients}/{total_clients} clientes...")

    all_clients_recommendations_df = pd.DataFrame(all_recommendations_list)
    if not all_clients_recommendations_df.empty:
        all_clients_recommendations_df['recommendation_rank'] = all_clients_recommendations_df.groupby('client')['recommendation_score'].rank(method='first', ascending=False).astype(int)
        all_clients_recommendations_df = all_clients_recommendations_df.sort_values(['client', 'recommendation_rank'])
        print(f"\nSe generaron {len(all_clients_recommendations_df)} filas de recomendación en total (filtros aplicados).")
    else:
        print("\nAdvertencia: No se generaron recomendaciones.")


    # --- Guardar Artefactos ---
    print("\nGuardando artefactos del modelo y recomendaciones pre-calculadas...")
    try:
        # Solo guardar si las matrices no son None
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

        if not all_clients_recommendations_df.empty:
            all_clients_recommendations_df.to_parquet(precomputed_recs_file, index=False)
            print(f"  - Recomendaciones pre-calculadas (filtradas) guardadas en: {precomputed_recs_file}")
        else:
            print("  - No se guardó archivo de recomendaciones pre-calculadas (estaba vacío).")

        print("\n--- Proceso de Entrenamiento y Pre-cálculo (Filtros aplicados) Finalizado ---")

    except Exception as e:
        print(f"Error guardando los artefactos: {e}")

    print(f"Tiempo requerido: {round(time.time()-start_time, 3)} seg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena recomendador Item-Item CF (filtrado) y pre-calcula recomendaciones.")
    parser.add_argument(
        '--config',
        default='params.yaml',
        help='Ruta al archivo de configuración params.yaml'
    )
    args = parser.parse_args()
    train_and_save(args.config)