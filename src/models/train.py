#!/usr/bin/env python3
"""
Entrena la regresión logística y guarda:
  • modelo          → models/logreg.joblib
  • máscara valid   → data/interim/valid_mask.parquet
  • métricas legibles→ reports/train_metrics.txt
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from dvclive import Live
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_split(df: pd.DataFrame, max_hist: str, valid_days: int):
    max_hist = pd.to_datetime(max_hist)
    train_mask = df.date <= max_hist - pd.Timedelta(days=valid_days)
    valid_mask = (df.date > max_hist - pd.Timedelta(days=valid_days)) & (
        df.date <= max_hist
    )

    feats = [
        "dow",
        "dom",
        "month",
        "is_weekend",
        "is_quincena",
        "days_since_last",
    ] + [c for c in df.columns if c.startswith("cnt_")]

    return (
        df.loc[train_mask, feats],
        df.loc[train_mask, "purchased"],
        df.loc[valid_mask, feats],
        df.loc[valid_mask, "purchased"],
        valid_mask,
    )


def main(args):
    # ── lee parámetros globales ─────────────────────────────────────
    with open("params.yaml") as f:
        prm = yaml.safe_load(f)
    pm, ps = prm["model"], prm["split"]

    # ── datos ───────────────────────────────────────────────────────
    cal = pd.read_parquet(args.calendar)
    _ = pd.read_json(args.features)  # compat
    X_tr, y_tr, X_val, y_val, val_mask = make_split(
        cal, ps["max_hist"], ps["valid_window_days"]
    )

    # ── modelo ──────────────────────────────────────────────────────
    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver=pm["solver"],
            max_iter=pm["max_iter"],
            tol=pm["tol"],
            C=pm["C"],
            class_weight=pm["class_weight"],
            n_jobs=-1,
        ),
    )
    pipe.fit(X_tr, y_tr)

    y_prob = pipe.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= pm["threshold"]).astype(int)

    # ── métricas ────────────────────────────────────────────────────
    auc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=3)

    # ── guarda artefactos ───────────────────────────────────────────
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        f.write(f"ROC-AUC: {auc:.3f}\n")
        f.write(f"Brier : {brier:.3f}\n")
        f.write("Confusion\n")
        f.write(f"{cm.tolist()}\n\n")
        f.write(report)

    pd.Series(val_mask, name="mask").to_frame().to_parquet(args.valid_mask_out)

    # ── DVC Live ────────────────────────────────────────────────────
    with Live() as live:
        live.log_metric("auc", auc)
        live.log_metric("brier", brier)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calendar", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--metrics_out", required=True)
    p.add_argument("--valid_mask_out", required=True)
    main(p.parse_args())
