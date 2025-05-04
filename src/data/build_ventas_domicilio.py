#!/usr/bin/env python3
r"""
Genera `ventas_domicilio_filtered.parquet` uniendo:

  • ventas_completo.csv               (obligatorio)
  • ventas_backtesting_exploted.csv   (opcional)

Ejemplo de uso
--------------
# con backtesting
python3 src/data/build_ventas_domicilio.py \
        --main data/raw/ventas_completo.csv \
        --backtest data/raw/ventas_backtesting_exploted.csv \
        --output data/raw/ventas_domicilio_filtered.parquet

# sin backtesting
python3 src/data/build_ventas_domicilio.py \
        --main data/raw/ventas_completo.csv \
        --output data/raw/ventas_domicilio_filtered.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

COLS_KEEP = [
    "date_sale",
    "id_point_sale",
    "identification_doct",
    "product",
    "invoice_value_with_discount_and_without_iva",
    "domicilio_status",
    "amount",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    """Carga un CSV con los dtypes correctos."""
    return pd.read_csv(
        path,
        usecols=COLS_KEEP,
        dtype={
            "identification_doct": str,
            "product": str,
            "domicilio_status": str,
        },
    )


def concat_sources(main_csv: Path, back_csv: Path | None) -> pd.DataFrame:
    print("→ Cargando ventas principales …")
    df = load_csv(main_csv)

    """if back_csv and back_csv.exists():
        print("→ Cargando ventas de backtesting …")
        df_back = load_csv(back_csv)
        df = (
            pd.concat([df, df_back], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"Total filas tras concatenar: {len(df):,}")
    else:
        if back_csv:
            print(f"⚠  Archivo backtesting no encontrado → se ignora: {back_csv}")
        print(f"Filas cargadas: {len(df):,}")"""

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra, limpia y agrupa sin lanzar SettingWithCopyWarning."""
    # id_client numérico y sin cero inicial
    df["id_client"] = df["identification_doct"].str.strip()
    df = df[
        df["id_client"].str.isdigit()
        & ~df["id_client"].str.startswith("0", na=False)
    ].copy()
    print(f"Filas tras filtrar id_client: {len(df):,}")

    # Limpieza básica (usar .loc para evitar el warning)
    df.loc[:, "product"] = df["product"].str.strip()
    df.loc[:, "date_sale"] = (
        pd.to_datetime(df["date_sale"], errors="coerce", dayfirst=True)
        .dt.normalize()
    )
    df = df.dropna(subset=["date_sale"])

    # Clientes con al menos un domicilio
    dom_true = df["domicilio_status"].str.lower().isin(["true", "1"])
    clients_dom = df.loc[dom_true, "id_client"].unique()
    df = df[df["id_client"].isin(clients_dom)]
    print(
        f"Clientes con domicilio: {len(clients_dom):,}  |  Filas: {len(df):,}"
    )

    # Renombrar y seleccionar
    df = df.rename(
        columns={
            "invoice_value_with_discount_and_without_iva": "amount_paid",
            "amount": "quantity",
        }
    )
    df = df.loc[
        :, ["date_sale", "id_client", "product", "quantity", "amount_paid"]
    ]

    # Agrupar por día‑cliente‑producto
    df = (
        df.groupby(["date_sale", "id_client", "product"], as_index=False)
        .agg(quantity=("quantity", "sum"), amount_paid=("amount_paid", "sum"))
        .sort_values(["id_client", "date_sale"])
    )

    print(f"Fecha máxima en datos resultantes: {df['date_sale'].max():%Y-%m-%d}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    try:
        raw_df = concat_sources(
            Path(args.main),
            Path(args.backtest) if args.backtest else None,
        )
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    clean_df = clean_dataframe(raw_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(args.output, index=False)
    print(f"✔ Parquet guardado en {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera ventas_domicilio_filtered.parquet"
    )
    parser.add_argument("--main", required=True, help="ventas_completo.csv")
    parser.add_argument(
        "--backtest",
        help="ventas_backtesting_exploted.csv (opcional, se ignora si no existe)",
    )
    parser.add_argument("--output", required=True, help="Ruta de salida .parquet")
    main(parser.parse_args())
