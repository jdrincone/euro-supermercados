# src/load_data.py
import pandas as pd
import yaml
from pathlib import Path
import argparse
from datetime import timedelta

def load_and_clean(config_path):
    """Carga, limpia y filtra las ventas de los últimos 6 meses."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    base_path = Path(data_params['base_path'])
    raw_path = base_path / data_params['raw_folder']
    processed_path = base_path / data_params['processed_folder']
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Cargando productos...")
    productos = pd.read_csv(
        raw_path / data_params['products_file'],
        dtype={"codigo_unico": str}
    ).rename(columns={"codigo_unico": "product"})
    productos = productos.loc[:, ["product", "description", "brand", "category"]]
    productos = productos.drop_duplicates("product")
    productos['product'] = productos['product'].str.strip()

    print("Cargando terceros...")
    terceros = pd.read_csv(
        raw_path / data_params['terceros_file'],
        converters={"document": str}
    )
    terceros = terceros.drop_duplicates("document")
    terceros = terceros.loc[:, ["document", "email", "telephone", "name"]]
    terceros['document'] = terceros['document'].str.strip()

    print("Cargando ventas...")
    cols = config['load_data']['sales_columns']
    dtype_sales = config['load_data']['sales_dtype']

    ventas_list = []
    for fname in data_params['sales_files']:
        print(f"  Leyendo {fname}...")
        df_v = pd.read_csv(
            raw_path / fname,
            usecols=cols,
            dtype=dtype_sales
        )
        df_v['identification_doct'] = df_v['identification_doct'].str.strip()
        df_v['product'] = df_v['product'].str.strip()
        df_v['date_sale'] = pd.to_datetime(
            df_v['date_sale'],
            errors='coerce',
        )
        ventas_list.append(df_v)

    ventas = pd.concat(ventas_list)
    print(f"Total ventas cargadas: {len(ventas)}")

    # Renombrar y filtrar IDs de cliente
    ventas['id_client'] = ventas['identification_doct'].astype(str).str.strip()
    mask_digits = ventas["id_client"].str.isdigit().fillna(False)
    mask_zero = ~ventas["id_client"].str.startswith("0", na=False)
    df = ventas[mask_digits & mask_zero].copy()
    print(f"Filas tras filtrar id_client no numéricos/cero inicial: {len(df)}")

    # Limpiar fechas y productos
    df['product'] = df['product'].astype(str).str.strip()
    df['date_sale'] = pd.to_datetime(df['date_sale']) # dayfirst ya aplicado
    df = df.dropna(subset=['date_sale'])
    df['date_sale'] = df['date_sale'].dt.normalize()

    # --- Filtrar ventas por los últimos 6 meses ---
    print("Filtrando ventas de los últimos 6 meses...")
    max_date = df['date_sale'].max()
    six_months_ago = max_date - timedelta(days=6 * 30)  # Aproximación de 6 meses
    df_filtered = df[df['date_sale'] >= six_months_ago].copy()
    print(f"Filas tras filtrar por los últimos 6 meses (desde {six_months_ago} hasta {max_date}): {len(df_filtered)}")

    # Seleccionar y guardar columnas relevantes
    output_cols = config['load_data']['output_columns']
    df_out = df_filtered[output_cols].copy()
    output_file = processed_path / config['load_data']['output_file']
    df_out.to_parquet(output_file, index=False)
    print(f"Datos iniciales (últimos 6 meses) guardados en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    load_and_clean(args.config)