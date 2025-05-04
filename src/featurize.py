# src/featurize.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
from datetime import timedelta

def create_features(config_path):
    """Crea el calendario completo y genera características temporales y lagged."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    featurize_params = config['featurize']
    base_path = Path(data_params['base_path'])
    processed_path = base_path / data_params['processed_folder']

    input_file = processed_path / config['preprocess']['daily_output_file']
    daily = pd.read_parquet(input_file)
    print(f"Datos diarios cargados desde: {input_file}, Filas: {len(daily)}")

    min_date = daily['date'].min()
    max_hist_date = daily['date'].max()
    future_days_offset = featurize_params['future_days_offset']
    max_calendar_date = max_hist_date + timedelta(days=future_days_offset)

    print(f"Creando calendario desde {min_date.date()} hasta {max_calendar_date.date()}")
    full_idx = pd.MultiIndex.from_product(
        [daily['client'].unique(),
         pd.date_range(min_date, max_calendar_date, freq='D')],
        names=['client', 'date']
    )

    calendar = (daily.set_index(['client', 'date'])
                .reindex(full_idx, fill_value=0)
                .reset_index())

    print("Generando características de fecha...")
    calendar['dow'] = calendar['date'].dt.dayofweek
    calendar['dom'] = calendar['date'].dt.day
    calendar['month'] = calendar['date'].dt.month
    calendar['is_weekend'] = calendar['dow'].isin([5, 6]).astype(int)
    quincena_days = featurize_params['quincena_days']
    calendar['is_quincena'] = (calendar['dom'].isin(quincena_days)).astype(int)

    print("Generando característica de recencia (days_since_last)...")
    calendar.sort_values(['client', 'date'], inplace=True)
    calendar['prev_buy'] = (
        calendar.groupby('client', group_keys=False)['date']
        .apply(lambda s: s.where(calendar.loc[s.index, 'purchased'].eq(1)).ffill())
    )
    # Rellenar NaT en prev_buy con una fecha muy anterior para evitar errores en resta
    min_allowable_date = calendar['date'].min() - timedelta(days=featurize_params['recency_fillna_days'] + 1)
    calendar['prev_buy'] = calendar['prev_buy'].fillna(min_allowable_date)

    calendar['days_since_last'] = (
        (calendar['date'] - calendar['prev_buy']).dt.days
    )
    # Asegurarse de que el fillna original (9999) se aplique correctamente
    calendar.loc[calendar['prev_buy'] == min_allowable_date, 'days_since_last'] = featurize_params['recency_fillna_days']
    calendar.drop(columns='prev_buy', inplace=True)


    print("Generando características de ventanas móviles (lagged counts)...")
    windows = featurize_params['rolling_windows']
    for w in windows:
        calendar[f'cnt_{w}d'] = (calendar.groupby('client')['purchased']
                                 .transform(lambda x: x.rolling(w, min_periods=1).sum().shift(1).fillna(0)))

    print(f"Características generadas: {list(calendar.columns)}")

    # Guardar calendario con características
    output_file = processed_path / config['featurize']['output_file']
    calendar.to_parquet(output_file, index=False)
    print(f"Calendario con características guardado en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    create_features(args.config)