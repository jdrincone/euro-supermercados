#!/usr/bin/env python3
"""Carga y limpieza inicial de los tres datasets crudos.

*Lee paths desde CLI (inyectados por DVC)
*Normaliza columnas clave
*Guarda un Parquet consolidado limpio (ventas)
"""
import argparse, pandas as pd, os


def load_and_clean(productos_path: str, terceros_path: str, ventas_path: str) -> pd.DataFrame:
    # Productos ───────────────────────────────────────────────────────────
    productos = (pd.read_csv(productos_path, dtype={"codigo_unico": str})
                   .rename(columns={"codigo_unico": "product"})
                   [["product", "description", "brand", "category"]]
                   .drop_duplicates("product"))
    productos["product"] = productos["product"].str.strip()

    # Clientes ────────────────────────────────────────────────────────────
    terceros = (pd.read_csv(terceros_path, converters={"document": str})
                  .drop_duplicates("document")
                  [["document", "email", "telephone", "name"]])
    terceros["document"] = terceros["document"].str.strip()

    # Ventas ──────────────────────────────────────────────────────────────
    ventas = pd.read_parquet(ventas_path)
    if not pd.api.types.is_datetime64_any_dtype(ventas["date_sale"]):
        ventas["date_sale"] = pd.to_datetime(ventas["date_sale"], errors="coerce", dayfirst=True)

    # Normaliza ids
    ventas["id_client"] = ventas["id_client"].astype(str).str.strip()
    ventas["product"]   = ventas["product"].astype(str).str.strip()

    # (opcional) join extra info si fuera necesario
    # ventas = ventas.merge(productos, how="left", on="product")
    # ventas = ventas.merge(terceros, how="left", left_on="id_client", right_on="document")

    return ventas


def main(args):
    ventas_clean = load_and_clean(args.productos, args.terceros, args.ventas)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ventas_clean.to_parquet(args.out, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--productos", required=True)
    p.add_argument("--terceros",  required=True)
    p.add_argument("--ventas",    required=True)
    p.add_argument("--out",       required=True)
    main(p.parse_args())