#!/usr/bin/env python3
"""
Calibra un modelo ya entrenado usando Platt scaling (sigmoid)
sobre el mismo conjunto de validación temporal empleado en train.py.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV


def main(args):
    # ── parámetros globales ──────────────────────────────────────────
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    split_cfg = params["split"]          # max_hist y ventana de validación

    # ── artefactos de entrada ───────────────────────────────────────
    base_model = joblib.load(args.model_in)            # modelo sin calibrar
    df         = pd.read_parquet(args.calendar)        # calendar con target
    with open(args.features, "r") as fh:
        features = json.load(fh)                       # lista de columnas

    # ── mismo split temporal que en entrenamiento ───────────────────
    max_hist   = pd.to_datetime(split_cfg["max_hist"])
    valid_days = split_cfg["valid_window_days"]
    valid_mask = (
        (df["date"] > max_hist - pd.Timedelta(days=valid_days))
        & (df["date"] <= max_hist)
    )

    X_val = df.loc[valid_mask, features]
    y_val = df.loc[valid_mask, "purchased"]

    # ── calibración tipo Platt ───────────────────────────────────────
    calib_model = CalibratedClassifierCV(base_model, method="sigmoid")
    calib_model.fit(X_val, y_val)

    # ── guarda el modelo calibrado ──────────────────────────────────
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calib_model, args.model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibra un modelo con Platt scaling")
    parser.add_argument("--model_in",   required=True, help="Ruta al modelo base (joblib)")
    parser.add_argument("--calendar",   required=True, help="Parquet con features y target")
    parser.add_argument("--features",   required=True, help="JSON con la lista de columnas feature")
    parser.add_argument("--model_out",  required=True, help="Ruta donde guardar el modelo calibrado")
    main(parser.parse_args())
