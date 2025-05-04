#!/usr/bin/env python3
"""
Backtesting día a día entre backtest.start y backtest.end (params.yaml).

Calcula TP, FP, FN, precision, recall y F0.5 para cada fecha.
Produce:
  • reports/backtest_metrics.csv  ← para análisis automático
  • reports/backtest_metrics.txt  ← misma info en texto plano
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml


def main(args):
    # ── parámetros de configuración ──────────────────────────────────
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    bt_cfg = cfg["backtest"]
    threshold = cfg["model"]["threshold"]

    # ── artefactos de entrada ────────────────────────────────────────
    model    = joblib.load(args.model)
    calendar = pd.read_parquet(args.calendar)

    # Lista de columnas feature (JSON → list[str])
    with open(args.features, "r") as fh:
        feats = json.load(fh)

    # Ventas reales del periodo de backtesting
    ventas_bk = pd.read_csv(
        args.ventas_back,
        parse_dates=["date_sale"],
        dtype={"identification_doct": str, "product": str}
    )
    ventas_bk["client"] = ventas_bk["identification_doct"].str.strip()

    # Filtra solo clientes existentes en calendar
    known_clients = set(calendar["client"].unique())
    ventas_bk = ventas_bk[ventas_bk["client"].isin(known_clients)]

    # ── loop día a día ───────────────────────────────────────────────
    fechas   = pd.date_range(bt_cfg["start"], bt_cfg["end"], freq="D")
    records  = []

    for fecha in fechas:
        reales = set(ventas_bk.loc[ventas_bk["date_sale"] == fecha, "client"])

        df_dia = calendar[calendar["date"] == fecha]
        probs  = model.predict_proba(df_dia[feats])[:, 1]
        pred   = set(df_dia.loc[probs >= threshold, "client"])

        tp = len(reales & pred)
        fp = len(pred - reales)
        fn = len(reales - pred)

        precision = tp / (tp + fp) if tp + fp else 0
        recall    = tp / (tp + fn) if tp + fn else 0
        beta      = 0.5
        f05       = ( (1 + beta**2) * precision * recall /
                      (beta**2 * precision + recall) ) if precision + recall else 0

        records.append(dict(
            fecha=str(fecha.date()),
            TP=tp, FP=fp, FN=fn,
            Precision=round(precision, 3),
            Recall=round(recall, 3),
            F0_5=round(f05, 3)
        ))

    out_df = pd.DataFrame(records)

    # ── guarda CSV ───────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # ── guarda versión texto “bonita” ────────────────────────────────
    txt_path = Path(args.out).with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write("fecha,TP,FP,FN,Precision,Recall,F0_5\n")
        out_df.to_csv(f, index=False, header=False, float_format="%.3f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting diario del modelo calibrado")
    parser.add_argument("--model",       required=True, help="Modelo calibrado (joblib)")
    parser.add_argument("--calendar",    required=True, help="Parquet con features + target")
    parser.add_argument("--ventas_back", required=True, help="CSV de ventas reales para backtest")
    parser.add_argument("--features",    required=True, help="JSON con lista de columnas feature")
    parser.add_argument("--out",         required=True, help="Ruta CSV de salida")
    main(parser.parse_args())
