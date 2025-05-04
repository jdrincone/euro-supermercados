#!/usr/bin/env python3
"""
Genera la curva de calibración (10 bins, estrategia «uniform») y la guarda como PNG.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay


def main(args):
    # ── artefactos de entrada ───────────────────────────────────────
    model = joblib.load(args.model)
    df    = pd.read_parquet(args.calendar)

    # Lista de features
    with open(args.features, "r") as fh:
        feats = json.load(fh)

    # Se utilizan todas las filas con etiqueta disponible
    mask = df["purchased"].notna()
    X, y = df.loc[mask, feats], df.loc[mask, "purchased"]

    # ── curva de calibración ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_estimator(
        model,
        X,
        y,
        n_bins=10,
        strategy="uniform",
        name="Regresión logística",
        ax=ax,
    )
    ax.set_title("Curva de calibración (10 bins)")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # ── guardar figura ──────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera la curva de calibración del modelo")
    p.add_argument("--model",    required=True)
    p.add_argument("--calendar", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out",      required=True)
    main(p.parse_args())
