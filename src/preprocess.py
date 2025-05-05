# src/preprocess.py
import pandas as pd
import yaml
from pathlib import Path
import argparse
from datetime import datetime, timedelta


def preprocess_sales(config_path):
    """Filtra ventas de clientes activos en los últimos
     3 meses con patrones de compra."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    preprocess_params = config['preprocess']
    base_path = Path(data_params['base_path'])
    last_month = data_params.get('last_month_with_sale', 3)
    processed_path = base_path / data_params['processed_folder']

    input_file = processed_path / config['load_data']['output_file']
    df = pd.read_parquet(input_file)
    print(f"Datos cargados desde: {input_file}, Filas: {len(df)}")


    # --- Filtrado por Actividad Reciente ---
    print(f"Filtrando clientes con compras en los últimos {last_month} meses...")
    three_months_ago = datetime.now().date() - timedelta(days=last_month*30)
    recent_purchases = df[df["date_sale"].dt.date >= three_months_ago]["id_client"].unique()
    print(f"Clientes únicos con compras en los últimos {last_month} meses: {len(recent_purchases)}")
    df_recent = df[df["id_client"].isin(recent_purchases)].copy()
    print(f"Filas tras filtrar por actividad reciente: {len(df_recent)}")

    # --- Filtrado por Patrones de Compra ---
    print("Calculando y filtrando por patrones de compra...")
    daily_purchases = df_recent.drop_duplicates(subset=['id_client', 'date_sale'])[['id_client', 'date_sale']]
    daily_purchases = daily_purchases.sort_values(['id_client', 'date_sale'])
    daily_purchases['days_between_purchases'] = daily_purchases.groupby('id_client')['date_sale'].diff().dt.days
    patterns = daily_purchases.groupby('id_client').agg(
        last_purchase_date=('date_sale', 'max'),
        first_purchase_date=('date_sale', 'min'),
        purchase_days_count=('date_sale', 'nunique'),
        avg_days_between=('days_between_purchases', 'mean'),
        median_days_between=('days_between_purchases', 'median'),
        std_days_between=('days_between_purchases', 'std')
    ).reset_index()

    min_purchase_count = preprocess_params['min_purchase_count']
    max_median_days = preprocess_params['max_median_days_between']
    max_std_days = preprocess_params['max_std_days_between']

    patterns_filtered = patterns[patterns['purchase_days_count'] >= min_purchase_count].copy()
    print(f"Clientes con >= {min_purchase_count} días de compra: {len(patterns_filtered)}")
    patterns_filtered = patterns_filtered[patterns_filtered["median_days_between"] < max_median_days]
    print(f"Clientes con mediana <= {max_median_days} días entre compras: {len(patterns_filtered)}")
    patterns_filtered = patterns_filtered[patterns_filtered["std_days_between"] <= max_std_days]
    print(f"Clientes con std <= {max_std_days} días entre compras: {len(patterns_filtered)}")

    df_filtered_pattern = df_recent[df_recent["id_client"].isin(patterns_filtered["id_client"])]
    print(f"Filas tras filtrar por patrones en clientes recientes: {len(df_filtered_pattern)}")

    # --- Agregación Diaria por Producto ---
    print("Agregando ventas diarias por producto...")
    df_final = df_filtered_pattern.rename(columns={
        'invoice_value_with_discount_and_without_iva': 'amount_paid',
        'amount': 'quantity'
    })
    df_agg = df_final.groupby(['date_sale', 'id_client', 'product']).agg(
        {"quantity": "sum", "amount_paid": "sum"}
    ).reset_index()
    df_agg.sort_values(['id_client', 'date_sale'], inplace=True)

    # --- GUARDAR df_agg PARA RECOMENDACIONES ---
    df_recommendation_output_file = processed_path / preprocess_params[
        'recommendation_output_file']

    df_agg.to_parquet(df_recommendation_output_file, index=False)
    print(f"Datos agregados por producto/cliente/día guardados en: {df_recommendation_output_file}")
    print("Clientes Unicos:", df_agg['id_client'].nunique())
    print("Productos Unicos:", df_agg['product'].nunique())
    print("Maxima  Fecha:", df_agg['date_sale'].max())


    # --- Agregación Diaria por Cliente (para featurize) ---
    print("Agregando ventas diarias por cliente ...")
    daily = (df_agg.groupby(['id_client', 'date_sale'], as_index=False)
             .agg(qty_tot=('quantity', 'sum'),
                  amount_tot=('amount_paid', 'sum'),
                  skus=('product', 'nunique'))
             .assign(purchased=1)
             .rename(columns={'id_client': 'client', 'date_sale': 'date'}))

    # Guardar resultados
    daily_output_file = processed_path / config['preprocess']['daily_output_file']
    daily.to_parquet(daily_output_file, index=False)
    print(f"Datos diarios agregados guardados en: {daily_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    preprocess_sales(args.config)