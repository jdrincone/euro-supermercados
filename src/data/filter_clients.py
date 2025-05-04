#!/usr/bin/env python3
"""Filtra clientes objetivo según frecuencia / variedad de compras."""
import argparse, yaml, pandas as pd, os


def main(args):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    tc = params["target_clients"]

    ventas = pd.read_parquet(args.ventas)

    resumen = (ventas.groupby("id_client")
                     .agg(num_fechas=("date_sale", "nunique"),
                          num_productos=("product", "nunique"))
                     .reset_index())

    clientes_ok = resumen[(resumen.num_fechas > tc["min_fechas"]) &
                          (resumen.num_productos > tc["min_productos"])]

    ventas_target = ventas[ventas.id_client.isin(clientes_ok.id_client)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ventas_target.to_parquet(args.out, index=False)

    print("Clientes objetivo:", clientes_ok.shape[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ventas", required=True)
    p.add_argument("--out",    required=True)
    main(p.parse_args())