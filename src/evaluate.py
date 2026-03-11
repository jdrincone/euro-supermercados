#!/usr/bin/env python3
"""Etapa DVC ``evaluate`` — evalúa, calibra el modelo y genera reportes.

Produce:
    - Modelo calibrado (``calibrated_model.joblib``).
    - Métricas JSON (ROC-AUC, Brier, Precision, Recall, F0.5).
    - Reportes de clasificación (base y calibrado).
    - Gráficos: curva de calibración, importancia de features, SHAP.

Uso::

    python src/evaluate.py --config params.yaml
"""

import argparse
import logging
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, model_dir, plots_dir, reports_dir
from data_io import (
    load_calendar_features,
    save_json,
    save_model,
    save_text_report,
)

matplotlib.use("Agg")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Validación temporal
# ---------------------------------------------------------------------------


def _calibration_test_split(
    df: pd.DataFrame,
    split_days: int,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Divide la ventana de validación en calibración (primera mitad) y test (segunda mitad).

    Esto evita evaluar el modelo calibrado sobre los mismos datos con los que se calibró,
    lo que inflaría las métricas reportadas.
    """
    last_hist = (
        df.loc[df["purchased"] == 1, "date"].max()
        if (df["purchased"] == 1).any()
        else df["date"].max() - timedelta(days=split_days)
    )
    train_end = last_hist - timedelta(days=split_days)
    cal_end = train_end + timedelta(days=split_days // 2)

    cal_mask = (df["date"] > train_end) & (df["date"] <= cal_end)
    test_mask = (df["date"] > cal_end) & (df["date"] <= last_hist)

    logger.info(
        "Calibración: %s -> %s | %d filas",
        (train_end + timedelta(days=1)).date(),
        cal_end.date(),
        cal_mask.sum(),
    )
    logger.info(
        "Test: %s -> %s | %d filas",
        (cal_end + timedelta(days=1)).date(),
        last_hist.date(),
        test_mask.sum(),
    )

    X_cal = df.loc[cal_mask, features]
    y_cal = df.loc[cal_mask, target]
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target]

    return X_cal, y_cal, X_test, y_test


# ---------------------------------------------------------------------------
#  Métricas
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thr: float,
) -> dict[str, float]:
    """Calcula métricas de clasificación dado un umbral."""
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        f"precision@{thr}": precision_score(y_true, y_pred),
        f"recall@{thr}": recall_score(y_true, y_pred),
        f"f0.5@{thr}": fbeta_score(y_true, y_pred, beta=0.5),
    }


# ---------------------------------------------------------------------------
#  Gráficos
# ---------------------------------------------------------------------------


def _extract_inner_model(cal: CalibratedClassifierCV):
    """Extrae el modelo original del calibrador (compatible multi-versión sklearn).

    Maneja FrozenEstimator y devuelve el modelo real (Pipeline o estimador).
    """
    for attr in ("base_estimator_", "estimator"):
        if hasattr(cal, attr):
            obj = getattr(cal, attr)
            # Desempaquetar FrozenEstimator si es necesario
            if hasattr(obj, "estimator"):
                return obj.estimator
            return obj
    inner = cal.calibrated_classifiers_[0].estimator
    if hasattr(inner, "estimator"):
        return inner.estimator
    return inner


def _plot_calibration(
    base_pipe: Pipeline,
    cal_pipe: CalibratedClassifierCV,
    X: pd.DataFrame,
    y: pd.Series,
    bins: int,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_estimator(base_pipe, X, y, n_bins=bins, name="Base", ax=ax)
    CalibrationDisplay.from_estimator(
        cal_pipe, X, y, n_bins=bins, name="Calibrado", ax=ax
    )
    ax.set_title("Curva de Calibración")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Curva de calibración: %s", out)


def _plot_feature_importance(
    model,
    feature_names: list[str],
    out: Path,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> None:
    """Grafica importancia de features (compatible con LR y tree-based models).

    Para modelos lineales usa |coef|. Para tree-based usa permutation importance.
    """
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        xlabel = "Importancia |coef|"
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance

        # Muestra para no demorar demasiado
        n = min(5000, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), n, replace=False)
        result = permutation_importance(
            model, X.iloc[idx], y.iloc[idx], n_repeats=10, random_state=42
        )
        importance = result.importances_mean
        xlabel = "Permutation Importance"
    else:
        logger.warning("No se pudo calcular importancia de features. Omitiendo plot.")
        return

    top = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel(xlabel)
    ax.set_title("Top 15 Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Feature importance: %s", out)


def _plot_shap(
    scaler,
    logreg,
    X: pd.DataFrame,
    feat_names: list[str],
    sample: int,
    out: Path,
) -> None:
    try:
        import shap
    except (ImportError, TypeError) as exc:
        logger.warning("SHAP no disponible (%s). Omitiendo.", exc)
        return

    X_scaled = scaler.transform(X)
    if sample < len(X_scaled):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), min(sample, len(X_scaled)), replace=False)
        X_scaled = X_scaled[idx]
        logger.info("SHAP con muestra de %d registros", len(X_scaled))
    df_scaled = pd.DataFrame(X_scaled, columns=feat_names)

    explainer = shap.LinearExplainer(logreg, df_scaled)
    shap_vals = explainer.shap_values(df_scaled)

    fig = plt.figure()
    shap.summary_plot(shap_vals, df_scaled, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("SHAP summary: %s", out)


def _plot_precision_recall_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thr: float,
    out: Path,
) -> None:
    """Grafica curva Precision-Recall con el umbral actual marcado."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, linewidth=2, label="Modelo calibrado")
    ax.fill_between(rec, prec, alpha=0.1)

    # Marcar umbral actual
    idx = np.searchsorted(thresholds, thr)
    if idx < len(prec) - 1:
        ax.scatter(
            rec[idx],
            prec[idx],
            s=100,
            color="red",
            zorder=5,
            label=f"Umbral={thr} (P={prec[idx]:.2f}, R={rec[idx]:.2f})",
        )

    # Marcar umbrales alternativos
    for alt_thr in [0.30, 0.40, 0.60]:
        alt_idx = np.searchsorted(thresholds, alt_thr)
        if alt_idx < len(prec) - 1:
            ax.scatter(
                rec[alt_idx],
                prec[alt_idx],
                s=60,
                marker="^",
                zorder=4,
                label=f"thr={alt_thr} (P={prec[alt_idx]:.2f}, R={rec[alt_idx]:.2f})",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall (para selección de umbral)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Curva PR: %s", out)


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def evaluate_model(config_path: str | Path) -> None:
    """Evalúa el modelo base, lo calibra y genera todos los artefactos de reporte.

    Usa un split de 3 vías dentro de la ventana de validación:
    - Primera mitad: calibración (ajuste de probabilidades).
    - Segunda mitad: test (evaluación honesta, datos NO vistos por el calibrador).

    Esto evita la sobreestimación de métricas que ocurre cuando se calibra
    y evalúa sobre el mismo dataset.
    """
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    eval_cfg = cfg["evaluate"]
    rpt_cfg = cfg["reports"]
    thr = eval_cfg["evaluation_threshold"]

    # Cargar datos y modelo base
    calendar = load_calendar_features(cfg)
    base_model_path = model_dir(cfg) / cfg["model"]["model_name"]
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"Modelo base no encontrado: {base_model_path}. Ejecuta `dvc repro` primero."
        )
    base_pipe: Pipeline = joblib.load(base_model_path)
    logger.info("Modelo base cargado: %s", base_model_path)

    # Split de 3 vías: calibración (1ra mitad) + test (2da mitad)
    X_cal, y_cal, X_test, y_test = _calibration_test_split(
        calendar,
        train_cfg["split_days_validation"],
        train_cfg["target"],
        train_cfg["features"],
    )

    # Directorios de salida
    rpt_dir = reports_dir(cfg)
    plt_dir = plots_dir(cfg)
    rpt_dir.mkdir(parents=True, exist_ok=True)
    plt_dir.mkdir(parents=True, exist_ok=True)

    # Evaluación modelo base (sobre test set)
    logger.info("Evaluando modelo base sobre test set...")
    base_proba = base_pipe.predict_proba(X_test)[:, 1]
    base_metrics = _compute_metrics(y_test, base_proba, thr)
    save_text_report(
        rpt_dir / rpt_cfg["base_class_report_file"],
        "Classification Report (Base, test set)",
        classification_report(y_test, base_pipe.predict(X_test), digits=3),
    )

    # Calibración sobre cal set (FrozenEstimator: no re-entrena el modelo base)
    logger.info(
        "Calibrando (método=%s) sobre set de calibración...",
        eval_cfg["calibration_method"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(base_pipe),
            method=eval_cfg["calibration_method"],
        ).fit(X_cal, y_cal)

    save_model(calibrator, model_dir(cfg) / cfg["model"]["calibrated_model_name"])

    # Evaluación modelo calibrado (sobre test set — datos NO vistos por el calibrador)
    logger.info("Evaluando modelo calibrado sobre test set (holdout)...")
    cal_proba = calibrator.predict_proba(X_test)[:, 1]
    cal_metrics = _compute_metrics(y_test, cal_proba, thr)
    save_text_report(
        rpt_dir / rpt_cfg["calibrated_class_report_file"],
        f"Classification Report (Calibrado, thr={thr}, test set)",
        classification_report(y_test, (cal_proba >= thr).astype(int), digits=3),
    )

    # Métricas consolidadas
    metrics = {f"base_{k}": v for k, v in base_metrics.items()} | {
        f"cal_{k}": v for k, v in cal_metrics.items()
    }
    save_json(rpt_dir / rpt_cfg["metrics_file"], metrics)

    # Gráficos
    _plot_calibration(
        base_pipe,
        calibrator,
        X_test,
        y_test,
        eval_cfg["calibration_bins"],
        plt_dir / rpt_cfg["calibration_plot"],
    )
    _plot_precision_recall_curve(
        y_test,
        cal_proba,
        thr,
        plt_dir / "precision_recall_curve.png",
    )

    # Plots de interpretabilidad — usar base_pipe directamente (ya está cargado)
    feat_names = train_cfg["features"]
    importance_out = plt_dir / rpt_cfg["importance_plot"]
    shap_out = plt_dir / rpt_cfg["shap_summary_plot"]

    if hasattr(base_pipe, "named_steps"):
        # Pipeline (LogisticRegression): extraer scaler + modelo
        scaler = base_pipe.named_steps.get("standardscaler")
        logreg = base_pipe.named_steps.get("logisticregression")
        if logreg:
            _plot_feature_importance(logreg, feat_names, importance_out)
        if scaler and logreg:
            _plot_shap(
                scaler,
                logreg,
                X_test,
                feat_names,
                eval_cfg["shap_sample_size"],
                shap_out,
            )
    else:
        # Modelo tree-based: usar permutation importance
        _plot_feature_importance(base_pipe, feat_names, importance_out, X_test, y_test)

    # Asegurar que todos los outputs requeridos por DVC existen
    for required in (importance_out, shap_out):
        if not required.exists():
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "No disponible para este tipo de modelo",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            fig.savefig(required)
            plt.close(fig)
            logger.info("Placeholder generado: %s", required)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa y calibra el modelo.")
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
