# src/train.py
import pandas as pd
import yaml
from pathlib import Path
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import timedelta

def train_model(config_path):
    """Entrena el modelo de Regresión Logística."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    train_params = config['train']
    base_path = Path(data_params['base_path'])
    processed_path = base_path / data_params['processed_folder']
    model_path = Path(config['model']['model_dir'])
    model_path.mkdir(parents=True, exist_ok=True)

    input_file = processed_path / config['featurize']['output_file']
    calendar = pd.read_parquet(input_file)
    print(f"Calendario cargado desde: {input_file}, Filas: {len(calendar)}")

    # Definir fechas de corte para train/validation
    max_hist = calendar[calendar['purchased'] == 1]['date'].max() # Usar la última compra real
    if pd.isna(max_hist): # Manejar caso sin compras
        max_hist = calendar['date'].max() - timedelta(days=train_params['split_days_validation'])

    train_end_date = max_hist - timedelta(days=train_params['split_days_validation'])
    valid_end_date = max_hist

    print(f"Fecha máxima histórica de compra usada para corte: {max_hist.date()}")
    print(f"Datos de entrenamiento hasta: {train_end_date.date()}")
    print(f"Datos de validación desde {train_end_date.date() + timedelta(days=1)} hasta {valid_end_date.date()}")

    train_mask = calendar["date"] <= train_end_date
    valid_mask = (calendar["date"] > train_end_date) & (calendar["date"] <= valid_end_date)

    features = train_params['features']
    target = train_params['target']

    X_train, y_train = calendar.loc[train_mask, features], calendar.loc[train_mask, target]
    X_valid, y_valid = calendar.loc[valid_mask, features], calendar.loc[valid_mask, target]

    print(f"Tamaño Train: {X_train.shape}, Target: {y_train.mean():.3f}")
    print(f"Tamaño Valid: {X_valid.shape}, Target: {y_valid.mean():.3f}")

    # Definir y entrenar pipeline
    print("Entrenando pipeline (Scaler + LogisticRegression)...")
    lr_params = train_params['logistic_regression']
    pipe = make_pipeline(
        StandardScaler(with_mean=False), # Mantener with_mean=False como en el notebook
        LogisticRegression(
            solver=lr_params['solver'],
            max_iter=lr_params['max_iter'],
            tol=lr_params['tol'],
            C=lr_params['C'],
            class_weight=lr_params['class_weight'],
            random_state=config['base']['random_state'], # Añadir random_state para reproducibilidad
            n_jobs=-1
        )
    )
    pipe.fit(X_train, y_train)
    print(f"Iteraciones Logistic Regression: {pipe.named_steps['logisticregression'].n_iter_[0]}")

    # Guardar modelo
    model_file = model_path / config['model']['model_name']
    joblib.dump(pipe, model_file)
    print(f"Modelo (pipeline) guardado en: {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    train_model(args.config)