import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml  # Asegúrate de tener PyYAML instalado: pip install PyYAML


# --- Funciones Auxiliares ---
def get_first_mode(series):
    mode_series = series.mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    return np.nan


# --- Funciones de Carga y Preprocesamiento ---
def load_sales_data(file_path, id_col='identification_doct', product_col='product',
                    date_col='date_sale', delivery_status_col='domicilio_status'):
    print(f"Cargando datos de ventas desde: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: El archivo de ventas '{file_path}' no fue encontrado.")
        raise
    df['id_client'] = df[id_col].astype(str).str.strip()
    mask_digits = df["id_client"].str.isdigit().fillna(False)
    mask_zero = ~df["id_client"].str.startswith("0", na=False)
    df = df[mask_digits & mask_zero].copy()
    print(f"Filas tras filtrar id_client: {len(df)}")

    df['product'] = df[product_col].astype(str).str.strip()
    df['date_sale'] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=['date_sale'], inplace=True)
    df['date_sale'] = df['date_sale'].dt.normalize()
    df['domicilio_status'] = (
        df[delivery_status_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq('true')
    ).astype('boolean')
    return df


def load_product_data(file_path, product_code_col='codigo_unico', product_desc_col='description'):
    print(f"Cargando datos de productos desde: {file_path}")
    try:
        df = pd.read_csv(file_path, dtype={product_code_col: str})
        df = df.rename(columns={product_code_col: "product"})
    except FileNotFoundError:
        print(f"Error: El archivo de productos '{file_path}' no fue encontrado.")
        raise
    # Asegurarse que las columnas esperadas existan, si no, crear vacías o manejar el error
    expected_cols = ["product", product_desc_col, "brand", "category"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA  # o None, o manejar el error

    df = df.loc[:, ["product", product_desc_col, "brand", "category"]]  # Usar product_desc_col
    df.drop_duplicates("product", inplace=True)
    df['product'] = df['product'].str.strip()
    print(f"Productos únicos cargados: {len(df)}")
    return df


# --- Funciones de Lógica de Negocio y Modelado ---
def filter_recent_customers(df_sales, months_filter):
    print(f"Filtrando clientes con compras en los últimos {months_filter} meses...")
    cutoff_date = datetime.now().date() - timedelta(days=months_filter * 30)
    if df_sales['date_sale'].dt.tz is not None:
        df_sales['date_sale_date_only'] = df_sales['date_sale'].dt.tz_localize(None).dt.date
        recent_customer_ids = df_sales[df_sales["date_sale_date_only"] >= cutoff_date]["id_client"].unique()
        df_sales.drop(columns=['date_sale_date_only'], inplace=True)
    else:
        recent_customer_ids = df_sales[df_sales["date_sale"].dt.date >= cutoff_date]["id_client"].unique()
    print(f"Clientes únicos con compras recientes: {len(recent_customer_ids)}")
    df_recent = df_sales[df_sales["id_client"].isin(recent_customer_ids)].copy()
    print(f"Filas tras filtrar por actividad reciente: {len(df_recent)}")
    return df_recent


def calculate_and_filter_purchase_patterns(df_recent, min_purchase_days, max_median_days, max_std_days):
    print("Calculando y filtrando por patrones de compra...")
    cols_for_patterns = [
        'id_client', 'date_sale', 'id_point_sale',
        'invoice_value_with_discount_and_without_iva', 'domicilio_status', 'product'
    ]
    existing_cols = [col for col in cols_for_patterns if col in df_recent.columns]
    daily_purchases = df_recent.drop_duplicates(subset=['id_client', 'date_sale'])[existing_cols]
    daily_purchases = daily_purchases.sort_values(['id_client', 'date_sale'])
    daily_purchases['days_between_purchases'] = daily_purchases.groupby('id_client')['date_sale'].diff().dt.days

    patterns_df = daily_purchases.groupby('id_client').agg(
        purchase_days_count=('date_sale', 'nunique'),
        count_products=('product', 'nunique'),  # 'nunique' para productos únicos comprados por el cliente
        median_days_between=('days_between_purchases', 'median'),
        std_days_between=('days_between_purchases', 'std'),
        pay_amount_mean=("invoice_value_with_discount_and_without_iva", "mean"),
        # Opcional: añadir otras features si son necesarias para el clustering
        # last_purchase_date=('date_sale', 'max'),
        # first_purchase_date=('date_sale', 'min'),
        # avg_days_between=('days_between_purchases', 'mean'),
        # most_frequent_point_sale=("id_point_sale", get_first_mode),
        # most_frequent_domicilio_status=("domicilio_status", get_first_mode)
    ).reset_index()

    patterns_filtered_df = patterns_df[patterns_df['purchase_days_count'] >= min_purchase_days].copy()
    print(f"Clientes con >= {min_purchase_days} días de compra: {len(patterns_filtered_df)}")
    patterns_filtered_df = patterns_filtered_df[patterns_filtered_df["median_days_between"] < max_median_days]
    print(f"Clientes con mediana <= {max_median_days} días entre compras: {len(patterns_filtered_df)}")
    patterns_filtered_df = patterns_filtered_df[patterns_filtered_df["std_days_between"] <= max_std_days]
    print(f"Clientes con std <= {max_std_days} días entre compras: {len(patterns_filtered_df)}")
    return patterns_filtered_df, existing_cols


def perform_customer_clustering(patterns_filtered_df, features_for_clustering, n_clusters, random_state_kmeans):
    if patterns_filtered_df.empty:
        print("No hay clientes para clusterizar.")
        return patterns_filtered_df, pd.DataFrame()  # Devuelve DataFrames vacíos

    print(f"\n--- Iniciando Clustering de Clientes ---")
    print(f"Clientes a clusterizar: {len(patterns_filtered_df)}")

    X_clustering = patterns_filtered_df[features_for_clustering].copy()
    X_clustering.fillna(X_clustering.mean(), inplace=True)

    if X_clustering.empty or X_clustering.isnull().any().any():
        print("Datos para clustering vacíos o con NaNs. No se puede proceder.")
        return patterns_filtered_df, pd.DataFrame()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state_kmeans, n_init='auto')
    patterns_filtered_df['purchase_pattern_cluster'] = kmeans.fit_predict(X_scaled)

    print("\n--- Características de los Clusters ---")
    cluster_summary = patterns_filtered_df.groupby('purchase_pattern_cluster').agg(
        count_clients=('id_client', 'count'),
        avg_median_days=('median_days_between', 'mean'),
        avg_std_days=('std_days_between', 'mean'),
        avg_count_products=('count_products', 'mean'),
        avg_pay_amount_mean=('pay_amount_mean', 'mean'),
    ).reset_index()
    print(cluster_summary)
    return patterns_filtered_df, cluster_summary


def generate_recommendations_by_cluster(df_sales_pattern_filtered, patterns_with_clusters_df,
                                        productos_df, cluster_summary_df,
                                        excluded_product_descriptions, product_desc_col='description'):
    print("\n--- Generando Recomendaciones por Cluster ---")
    df_sales_with_clusters = pd.merge(
        df_sales_pattern_filtered[['id_client', 'product']],
        patterns_with_clusters_df[['id_client', 'purchase_pattern_cluster']],
        on='id_client', how='left'
    )
    df_sales_with_clusters = pd.merge(
        df_sales_with_clusters,
        productos_df[['product', product_desc_col]],  # Usar product_desc_col
        on='product', how='left'
    )
    df_sales_with_clusters = df_sales_with_clusters[
        ~df_sales_with_clusters[product_desc_col].isin(excluded_product_descriptions)  # Usar product_desc_col
    ]
    df_sales_with_clusters.dropna(subset=[product_desc_col], inplace=True)  # Usar product_desc_col

    cluster_recommendations_map = {}
    if patterns_with_clusters_df.empty or 'purchase_pattern_cluster' not in patterns_with_clusters_df.columns:
        print("No hay clusters para generar recomendaciones.")
        return cluster_recommendations_map

    for cluster_id in sorted(patterns_with_clusters_df['purchase_pattern_cluster'].unique()):
        #TODO: Para cada grupo, asignamos el TOP como el int( del valor promedio de compras)
        top_n_for_cluster = int(round(
                cluster_summary_df.loc[
                    cluster_summary_df['purchase_pattern_cluster'] == cluster_id, 'avg_count_products'].values[0]
            ))
        clients_in_cluster = patterns_with_clusters_df[
            patterns_with_clusters_df['purchase_pattern_cluster'] == cluster_id
            ]['id_client']
        sales_cluster_data = df_sales_with_clusters[
            df_sales_with_clusters['id_client'].isin(clients_in_cluster)
        ]
        if not sales_cluster_data.empty and product_desc_col in sales_cluster_data.columns:  # Usar product_desc_col
            top_products = sales_cluster_data[product_desc_col].value_counts().nlargest(
                top_n_for_cluster).index  # Usar product_desc_col
            cluster_recommendations_map[cluster_id] = top_products.tolist()
        else:
            cluster_recommendations_map[cluster_id] = []
        print(
            f"Cluster {cluster_id}: {len(cluster_recommendations_map.get(cluster_id, []))} productos recomendados (top_n={top_n_for_cluster})")
    return cluster_recommendations_map


def create_customer_recommendation_df(patterns_with_clusters_df, cluster_recommendations_map):
    print("\n--- Creando DataFrame de Recomendaciones por Cliente ---")
    if patterns_with_clusters_df.empty:
        print("No hay datos de patrones para crear el DataFrame de recomendaciones.")
        return pd.DataFrame(columns=['id_client', 'purchase_pattern_cluster', 'recommended_products'])

    df_client_recs = patterns_with_clusters_df[['id_client', 'purchase_pattern_cluster']].copy()
    df_client_recs['recommended_products'] = df_client_recs[
        'purchase_pattern_cluster'
    ].apply(lambda cid: cluster_recommendations_map.get(cid, []))
    print("\nDataFrame de Recomendaciones por Cliente (muestra):")
    print(df_client_recs.head())
    return df_client_recs


# --- Funciones de Salida y Configuración ---
def save_to_parquet(df, output_file_path_str):
    try:
        output_file = Path(output_file_path_str)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"\nRecomendaciones guardadas exitosamente en: {output_file}")
    except Exception as e:
        print(f"Error al guardar el archivo Parquet: {e}")


def load_config(config_path="config.yaml"):
    """Carga la configuración desde un archivo YAML."""
    print(f"Cargando configuración desde: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: El archivo de configuración '{config_path}' no fue encontrado.")
        raise
    except yaml.YAMLError as e:
        print(f"Error al parsear el archivo YAML: {e}")
        raise


# --- Orquestador Principal ---
def main_pipeline(config):
    """Orquesta todo el flujo del pipeline usando la configuración."""
    print("Iniciando el pipeline de segmentación y recomendación...")

    base_path = Path(config['data']['base_path'])
    raw_folder = config['data']['raw_folder']
    processed_folder = config['data']['processed_folder']

    sales_file = base_path / raw_folder / config['data']['sales_file']
    products_file = base_path / raw_folder / config['data']['products_file']

    rec_cluster_config = config['recommendations_clustering']

    # Construcción de la ruta de salida para las recomendaciones de clustering
    output_recommendations_folder = base_path / processed_folder / rec_cluster_config['output_folder']
    output_dir = Path(rec_cluster_config['output_folder'])
    output_parquet_file = output_dir / rec_cluster_config['precomputed_cluster_recs_file']

    try:
        df_ventas = load_sales_data(sales_file)
        # Asumiendo que la columna de descripción en productos.csv es 'description'
        df_productos = load_product_data(products_file, product_desc_col='description')
    except Exception:
        print("Finalizando el pipeline debido a errores en la carga de datos.")
        return

    df_recent_sales = filter_recent_customers(df_ventas, rec_cluster_config['months_recent_activity'])
    if df_recent_sales.empty:
        print("No hay datos de clientes recientes. Guardando archivo vacío.")
        save_to_parquet(pd.DataFrame(columns=['id_client', 'purchase_pattern_cluster', 'recommended_products']),
                        output_parquet_file)
        return

    patterns_filtered_df, existing_cols_patterns = calculate_and_filter_purchase_patterns(
        df_recent_sales,
        rec_cluster_config['min_purchase_days_pattern'],
        rec_cluster_config['max_median_days_pattern'],
        rec_cluster_config['max_std_days_pattern']
    )

    if patterns_filtered_df.empty:
        print("No hay clientes que cumplan patrones. Guardando archivo vacío.")
        save_to_parquet(pd.DataFrame(columns=['id_client', 'purchase_pattern_cluster', 'recommended_products']),
                        output_parquet_file)
        return

    df_sales_pattern_filtered = df_recent_sales[df_recent_sales["id_client"].isin(patterns_filtered_df["id_client"])][
        existing_cols_patterns].copy()

    patterns_with_clusters_df, cluster_summary_df = perform_customer_clustering(
        patterns_filtered_df,
        rec_cluster_config['features_for_clustering'],
        rec_cluster_config['n_clusters'],
        config['base']['random_state']
    )

    if patterns_with_clusters_df.empty or 'purchase_pattern_cluster' not in patterns_with_clusters_df.columns:
        print("Clustering no produjo resultados. Guardando archivo vacío.")
        save_to_parquet(pd.DataFrame(columns=['id_client', 'purchase_pattern_cluster', 'recommended_products']),
                        output_parquet_file)
        return

    recommendations_map = generate_recommendations_by_cluster(
        df_sales_pattern_filtered,
        patterns_with_clusters_df,
        df_productos,
        cluster_summary_df,
        rec_cluster_config['excluded_product_descriptions'],
        product_desc_col='description'  # Asegurar que esta columna existe en df_productos
    )

    df_final_recommendations = create_customer_recommendation_df(
        patterns_with_clusters_df, recommendations_map
    )

    save_to_parquet(df_final_recommendations, output_parquet_file)
    print("\nPipeline completado.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de segmentación y recomendación por clustering.")
    parser.add_argument("--config", default="config.yaml", help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()

    try:
        configuration = load_config(args.config)
        main_pipeline(configuration)
    except Exception as e:
        print(f"Error fatal en el pipeline: {e}")