#!/usr/bin/env python3
"""Construye el calendario diario con variables temporales y rolling counts."""
import argparse, yaml, pandas as pd, numpy as np, json, os


def build_calendar(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    future_end = pd.to_datetime(params["future_end_date"])
    windows    = params["calendar_windows"]
    quincena   = set(params["is_quincena_days"])

    daily = (df.groupby(["id_client", "date_sale"], as_index=False)
                .agg(qty_tot=("quantity", "sum"),
                     amount_tot=("amount_paid", "sum"),
                     skus=("product", "nunique"))
                .assign(purchased=1)
                .rename(columns={"id_client": "client", "date_sale": "date"}))

    min_date = daily.date.min()
    all_dates = pd.date_range(min_date, future_end, freq="D")
    full_idx = pd.MultiIndex.from_product([daily.client.unique(), all_dates],
                                          names=["client", "date"])

    cal = (daily.set_index(["client", "date"])
               .reindex(full_idx, fill_value=0)
               .reset_index())

    # Temporal features
    cal["dow"] = cal.date.dt.dayofweek
    cal["dom"] = cal.date.dt.day
    cal["month"] = cal.date.dt.month
    cal["is_weekend"] = cal.dow.isin([5, 6]).astype(int)
    cal["is_quincena"] = cal.dom.isin(quincena).astype(int)

    # days_since_last purchase
    cal.sort_values(["client", "date"], inplace=True)
    last_buy = cal.groupby("client")["date"].apply(lambda s: s.where(cal.loc[s.index, "purchased"] == 1).ffill())
    cal["days_since_last"] = (cal.date - last_buy).dt.days.fillna(9999).astype(int)

    # rolling counts
    for w in windows:
        cal[f"cnt_{w}d"] = (cal.groupby("client")["purchased"]
                               .transform(lambda x: x.rolling(w, min_periods=1).sum().shift(1).fillna(0)))

    return cal


def main(args):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    fe = params["feature_engineering"]

    ventas_target = pd.read_parquet(args.ventas)
    calendar = build_calendar(ventas_target, fe)

    # Guarda calendar y lista de features
    os.makedirs(os.path.dirname(args.calendar), exist_ok=True)
    calendar.to_parquet(args.calendar, index=False)

    features = ["dow", "dom", "month", "is_weekend", "is_quincena", "days_since_last"] + \
               [c for c in calendar.columns if c.startswith("cnt_")]
    with open(args.features, "w") as f:
        json.dump(features, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ventas",    required=True)
    p.add_argument("--calendar",  required=True)
    p.add_argument("--features",  required=True)
    main(p.parse_args())