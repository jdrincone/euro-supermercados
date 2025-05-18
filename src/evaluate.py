#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/evaluate.py
===============

Evalúa y calibra el modelo entrenado, genera métricas,
reportes y gráficos (calibración, importancias y SHAP).

Uso
----
$ python src/evaluate.py --config params.yaml
"""

import argparse
import json
import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from utils import read_yaml


matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def load_data_and_model(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Pipeline]:
    processed_path = Path(cfg["data"]["base_path"]) / cfg["data"]["processed_folder"]
    calendar = pd.read_parquet(processed_path / cfg["featurize"]["output_file"])
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    calendar["client"] = calendar["client"].astype(str)

    model_path = (
        Path(cfg["model"]["model_dir"]) / cfg["model"]["model_name"]
    )
    model_pipe: Pipeline = joblib.load(model_path)

    logging.info("Calendario: %s filas | Modelo cargado de %s", len(calendar), model_path)
    return calendar, model_pipe


def temporal_validation_split(
    df: pd.DataFrame, split_days: int, target_col: str, features: list[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    last_hist = (
        df.loc[df["purchased"] == 1, "date"].max()
        if (df["purchased"] == 1).any()
        else df["date"].max() - timedelta(days=split_days)
    )
    train_end = last_hist - timedelta(days=split_days)
    valid_mask = (df["date"] > train_end) & (df["date"] <= last_hist)

    logging.info(
        "🔪 Ventana validación: %s → %s | Filas: %s",
        (train_end + timedelta(days=1)).date(),
        last_hist.date(),
        valid_mask.sum(),
    )
    return df.loc[valid_mask, features], df.loc[valid_mask, target_col]


def evaluate_predictions(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        f"precision@{threshold}": precision_score(y_true, y_pred),
        f"recall@{threshold}": recall_score(y_true, y_pred),
        f"f0.5@{threshold}": fbeta_score(y_true, y_pred, beta=0.5),
    }


def save_txt(path: Path, header: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(header + "\n" + "=" * len(header) + "\n" + content)
    logging.info("Guardado %s", path)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=4)
    logging.info("Métricas en %s", path)


# --------- NUEVO: helper seguro para extraer el pipeline -------------------
def _extract_pipeline_from_calibrator(cal: CalibratedClassifierCV) -> Pipeline:
    """
    Devuelve el Pipeline original (StandardScaler → LogisticRegression)
    sin importar la versión de scikit-learn.
    """
    if hasattr(cal, "base_estimator_"):          # >= 1.4
        return cal.base_estimator_
    if hasattr(cal, "estimator"):                # <= 1.3 (cv='prefit')
        return cal.estimator
    # Fallback para cv != 'prefit'
    return cal.calibrated_classifiers_[0].estimator


# ------------------------- plots ------------------------------------------
def plot_calibration(
    base_pipe: Pipeline,
    cal_pipe: CalibratedClassifierCV,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    bins: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_estimator(
        base_pipe, X_valid, y_valid, n_bins=bins, name="Base", ax=ax
    )
    CalibrationDisplay.from_estimator(
        cal_pipe, X_valid, y_valid, n_bins=bins, name="Calibrado", ax=ax
    )
    ax.set_title("Curva de Calibración")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Curva de calibración guardada en %s", out_path)


def plot_feature_importance(
    scaler, logreg, feature_names: list[str], out_path: Path
) -> None:
    coeff = np.abs(logreg.coef_[0])
    top = (
        pd.DataFrame({"feature": feature_names, "importance": coeff})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importancia |coef|")
    ax.set_title("Top 15 Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Feature importance guardada en %s", out_path)


def plot_shap(
    scaler, logreg, X_valid: pd.DataFrame, feat_names: list[str], sample: int, out_path: Path
) -> None:
    X_scaled = scaler.transform(X_valid)
    if sample < len(X_scaled):
        idx = np.random.choice(len(X_scaled), sample, replace=False)
        X_scaled = X_scaled[idx]
        logging.info("SHAP con muestra de %s registros", sample)
    df_scaled = pd.DataFrame(X_scaled, columns=feat_names)

    explainer = shap.LinearExplainer(logreg, df_scaled)
    shap_vals = explainer.shap_values(df_scaled)

    fig = plt.figure()
    shap.summary_plot(shap_vals, df_scaled, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("SHAP summary guardado en %s", out_path)


# --------------------------------------------------------------------------- #
# Evaluación principal
# --------------------------------------------------------------------------- #
def evaluate_model(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)
    cfg_train = cfg["train"]

    # --- Carga ------------------------------------------------------------
    calendar, base_pipe = load_data_and_model(cfg)
    X_valid, y_valid = temporal_validation_split(
        calendar,
        cfg_train["split_days_validation"],
        cfg_train["target"],
        cfg_train["features"],
    )

    # --- Directorios de salida -------------------------------------------
    reports_dir = Path(cfg["reports"]["reports_dir"])
    plots_dir = reports_dir / cfg["reports"]["plots_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    thr = cfg["evaluate"]["evaluation_threshold"]

    # --- Evaluación base --------------------------------------------------
    logging.info("⚖️  Evaluando modelo base…")
    base_metrics = evaluate_predictions(
        y_valid, base_pipe.predict_proba(X_valid)[:, 1], thr
    )
    save_txt(
        reports_dir / cfg["reports"]["base_class_report_file"],
        "Classification Report (Base)",
        classification_report(y_valid, base_pipe.predict(X_valid), digits=3),
    )

    # --- Calibración ------------------------------------------------------
    logging.info("Calibrando (método=%s)…", cfg["evaluate"]["calibration_method"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The `cv='prefit'` option"
        )
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        calibrator = CalibratedClassifierCV(
            base_pipe, method=cfg["evaluate"]["calibration_method"], cv="prefit"
        ).fit(X_valid, y_valid)

    joblib.dump(
        calibrator,
        Path(cfg["model"]["model_dir"]) / cfg["model"]["calibrated_model_name"],
    )

    # --- Evaluación calibrada --------------------------------------------
    logging.info("⚖️  Evaluando modelo calibrado…")
    cal_metrics = evaluate_predictions(
        y_valid,
        calibrator.predict_proba(X_valid)[:, 1],
        thr,
    )
    save_txt(
        reports_dir / cfg["reports"]["calibrated_class_report_file"],
        f"Classification Report (Calibrado, thr={thr})",
        classification_report(
            y_valid,
            (calibrator.predict_proba(X_valid)[:, 1] >= thr).astype(int),
            digits=3,
        ),
    )

    # --- Guardar métricas -------------------------------------------------
    metrics = {f"base_{k}": v for k, v in base_metrics.items()} | {
        f"cal_{k}": v for k, v in cal_metrics.items()
    }
    save_json(reports_dir / cfg["reports"]["metrics_file"], metrics)

    # --- Gráficos ---------------------------------------------------------
    plot_calibration(
        base_pipe,
        calibrator,
        X_valid,
        y_valid,
        cfg["evaluate"]["calibration_bins"],
        plots_dir / cfg["reports"]["calibration_plot"],
    )

    # -------- Extraer pipeline seguro y hacer plots de interpretabilidad --
    pipeline = _extract_pipeline_from_calibrator(calibrator)
    scaler = pipeline.named_steps["standardscaler"]
    logreg = pipeline.named_steps["logisticregression"]
    feat_names = cfg_train["features"]

    plot_feature_importance(
        scaler,
        logreg,
        feat_names,
        plots_dir / cfg["reports"]["importance_plot"],
    )
    plot_shap(
        scaler,
        logreg,
        X_valid,
        feat_names,
        cfg["evaluate"]["shap_sample_size"],
        plots_dir / cfg["reports"]["shap_summary_plot"],
    )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa y calibra el modelo.")
    parser.add_argument(
        "--config", default="params.yaml", help="Ruta al YAML de configuración."
    )
    args = parser.parse_args()
    evaluate_model(args.config)
