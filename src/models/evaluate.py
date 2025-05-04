#!/usr/bin/env python3
"""
Evalúa el modelo calibrado y guarda:
  • metrics.json (detallado para DVC diff)
  • metrics.txt  (legible: ROC‑AUC, Brier, confusion, report)
  • confusion_matrix.png
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
)


def main(args):
    # ── artefactos de entrada ───────────────────────────────────────
    model = joblib.load(args.model_path)
    df = pd.read_parquet(args.calendar_path)

    with open(args.feat_path) as fh:
        feats = json.load(fh)

    valid_mask = pd.read_parquet(args.split_mask_path)["mask"].values

    X_val = df.loc[valid_mask, feats]
    y_val = df.loc[valid_mask, "purchased"].values
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ── métricas principales ────────────────────────────────────────
    auc = roc_auc_score(y_val, y_prob)
    brier = brier_score_loss(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    logloss = log_loss(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=3)

    # Mejor F0.5
    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    beta = 0.5
    f05 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
    best_f05 = float(np.nanmax(f05))

    # ── metrics.json para DVC diff ──────────────────────────────────
    metrics_json = {
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "logloss": logloss,
        "best_f05": best_f05,
        "confusion_matrix": cm.tolist(),
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics_json, f, indent=2)

    # ── metrics.txt legible ─────────────────────────────────────────
    txt_path = Path(args.metrics_out).with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"ROC-AUC: {auc:.3f}\n")
        f.write(f"Brier : {brier:.3f}\n")
        f.write("Confusion\n")
        f.write(f"{cm.tolist()}\n\n")
        f.write(report)

    # ── figura de la matriz de confusión ────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{v:,}",
            ha="center",
            va="center",
            color="white" if im.norm(v) > 0.5 else "black",
        )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (no)", "1 (sí)"])
    ax.set_yticklabels(["0 (no)", "1 (sí)"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (0.5)")
    plt.tight_layout()
    Path(args.cm_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.cm_fig)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evalúa modelo calibrado")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calendar_path", required=True)
    p.add_argument("--feat_path", required=True)
    p.add_argument("--split_mask_path", required=True)
    p.add_argument("--metrics_out", required=True)
    p.add_argument("--cm_fig", required=True)
    main(p.parse_args())
