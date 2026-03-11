#!/usr/bin/env python3
"""Etapa DVC ``train`` — entrena modelo de clasificación binaria.

Soporta dos modelos configurables vía ``params.yaml``:

- ``logistic_regression``: StandardScaler → LogisticRegression (lineal, interpretable).
- ``hist_gradient_boosting``: HistGradientBoostingClassifier (no lineal, captura interacciones).

Flujo:
    1. Carga calendario de features.
    2. Split temporal train/valid.
    3. Entrena el modelo seleccionado.
    4. Guarda como ``model.joblib``.

Uso::

    python src/train.py --config params.yaml
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, model_dir, processed_path
from data_io import load_parquet, save_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Split temporal
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    split_days: int,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide datos en train/valid usando una ventana temporal.

    Usa la última fecha con compra como referencia. El set de validación
    cubre los últimos ``split_days`` días antes de esa fecha.
    """
    last_hist = (
        df.loc[df["purchased"] == 1, "date"].max()
        if (df["purchased"] == 1).any()
        else df["date"].max() - timedelta(days=split_days)
    )
    train_end = last_hist - timedelta(days=split_days)

    logger.info(
        "Split temporal | train <= %s | valid %s -> %s",
        train_end.date(),
        (train_end + timedelta(days=1)).date(),
        last_hist.date(),
    )

    train_mask = df["date"] <= train_end
    valid_mask = (df["date"] > train_end) & (df["date"] <= last_hist)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    X_valid = df.loc[valid_mask, features]
    y_valid = df.loc[valid_mask, target]

    logger.info(
        "Train %s | pos=%.3f — Valid %s | pos=%.3f",
        X_train.shape,
        y_train.mean(),
        X_valid.shape,
        y_valid.mean(),
    )
    return X_train, y_train, X_valid, y_valid


# ---------------------------------------------------------------------------
#  Builders de modelo
# ---------------------------------------------------------------------------


def _build_logistic_regression(lr_cfg: dict, seed: int):
    """Pipeline: StandardScaler → LogisticRegression."""
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            tol=lr_cfg["tol"],
            C=lr_cfg["C"],
            class_weight=lr_cfg["class_weight"],
            random_state=seed,
        ),
    )


def _build_hist_gradient_boosting(hgb_cfg: dict, seed: int):
    """HistGradientBoostingClassifier (no necesita scaler)."""
    return HistGradientBoostingClassifier(
        max_iter=hgb_cfg.get("max_iter", 200),
        max_depth=hgb_cfg.get("max_depth", 6),
        learning_rate=hgb_cfg.get("learning_rate", 0.1),
        min_samples_leaf=hgb_cfg.get("min_samples_leaf", 50),
        l2_regularization=hgb_cfg.get("l2_regularization", 0.1),
        class_weight="balanced",
        random_state=seed,
    )


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def train_model(config_path: str | Path) -> None:
    """Entrena el modelo configurado y lo guarda."""
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    seed = cfg["base"]["random_state"]

    proc = processed_path(cfg)
    calendar = load_parquet(proc / cfg["featurize"]["output_file"], "Calendario")

    X_train, y_train, _, _ = temporal_split(
        calendar,
        train_cfg["split_days_validation"],
        train_cfg["target"],
        train_cfg["features"],
    )

    model_type = train_cfg.get("model_type", "logistic_regression")

    if model_type == "hist_gradient_boosting":
        hgb_cfg = train_cfg.get("hist_gradient_boosting", {})
        model = _build_hist_gradient_boosting(hgb_cfg, seed)
        logger.info("Entrenando HistGradientBoostingClassifier...")
        model.fit(X_train, y_train)
        logger.info("Árboles entrenados: %d", model.n_iter_)
    else:
        lr_cfg = train_cfg["logistic_regression"]
        model = _build_logistic_regression(lr_cfg, seed)
        logger.info("Entrenando LogisticRegression...")
        model.fit(X_train, y_train)
        logger.info(
            "Iteraciones: %d", model.named_steps["logisticregression"].n_iter_[0]
        )

    model_out = model_dir(cfg) / cfg["model"]["model_name"]
    save_model(model, model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo de clasificación.")
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    train_model(args.config)
