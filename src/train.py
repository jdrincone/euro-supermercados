#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/train.py
============

Entrena un modelo de Regresión Logística con estandarización previa
usando el calendario de features creado en la fase *featurize*.

Flujo:
1. Lee `params.yaml`.
2. Carga el calendario (cliente-día) desde parquet.
3. Separa datos en *train* y *valid* según ventana temporal.
4. Entrena un pipeline: `StandardScaler(with_mean=False)` → `LogisticRegression`.
5. Guarda el pipeline (`joblib`) en la carpeta de modelos.

Uso
----
$ python src/train.py --config params.yaml
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import read_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------------------------------------------------------- #
#  Funciones auxiliares
# --------------------------------------------------------------------------- #


def load_calendar(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logging.info("Calendario cargado: %s filas | %s columnas", *df.shape)
    return df


def train_valid_split(
    df: pd.DataFrame, split_days: int, target_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Crea máscaras temporales para separar train/valid.

    * Usa la última fecha real de compra (`purchased == 1`) como referencia.
    * Si no existen compras, toma la fecha máxima absoluta.
    """
    last_hist = (
        df.loc[df["purchased"] == 1, "date"].max()
        if (df["purchased"] == 1).any()
        else df["date"].max() - timedelta(days=split_days)
    )

    train_end = last_hist - timedelta(days=split_days)
    valid_end = last_hist

    logging.info(
        "Corte temporal | train ≤ %s | valid %s → %s",
        train_end.date(),
        (train_end + timedelta(days=1)).date(),
        valid_end.date(),
    )

    train_mask = df["date"] <= train_end
    valid_mask = (df["date"] > train_end) & (df["date"] <= valid_end)

    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target_col]
    X_valid, y_valid = df.loc[valid_mask, features], df.loc[valid_mask, target_col]

    logging.info(
        "Train %s | Positives %.3f — Valid %s | Positives %.3f",
        X_train.shape,
        y_train.mean(),
        X_valid.shape,
        y_valid.mean(),
    )
    return X_train, y_train, X_valid, y_valid


def build_pipeline(lr_cfg: Dict[str, Any], seed: int):
    """Crea el pipeline Scaler → LogisticRegression con hiperparámetros del YAML."""
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            tol=lr_cfg["tol"],
            C=lr_cfg["C"],
            class_weight=lr_cfg["class_weight"],
            random_state=seed,
            n_jobs=-1,
        ),
    )
    return pipe


def save_model(pipe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    logging.info("Modelo guardado en %s", path.resolve())


# --------------------------------------------------------------------------- #
#  Entrenamiento principal
# --------------------------------------------------------------------------- #
def train_model(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)

    # --- Rutas ------------------------------------------------------------
    data_params = cfg["data"]
    train_params = cfg["train"]
    model_cfg = cfg["model"]

    processed_path = Path(data_params["base_path"]) / data_params["processed_folder"]
    calendar_in = processed_path / cfg["featurize"]["output_file"]
    model_out = Path(model_cfg["model_dir"]) / model_cfg["model_name"]

    # --- Datos ------------------------------------------------------------
    calendar = load_calendar(calendar_in)

    global features  # usado en train_valid_split
    features = train_params["features"]
    target = train_params["target"]

    # --- Split temporal ---------------------------------------------------
    X_train, y_train, X_valid, y_valid = train_valid_split(
        calendar, train_params["split_days_validation"], target
    )

    # --- Pipeline & fit ---------------------------------------------------
    pipe = build_pipeline(train_params["logistic_regression"], cfg["base"]["random_state"])
    logging.info("Entrenando LogisticRegression…")
    pipe.fit(X_train, y_train)
    logging.info(
        "Entrenado en %s iteraciones",
        pipe.named_steps["logisticregression"].n_iter_[0],
    )

    # --- Guardar ----------------------------------------------------------
    save_model(pipe, model_out)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de Regresión Logística."
    )
    parser.add_argument(
        "--config",
        default="params.yaml",
        help="Ruta al archivo YAML de configuración (default: params.yaml)",
    )
    args = parser.parse_args()
    train_model(args.config)
