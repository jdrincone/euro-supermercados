#!/usr/bin/env python3
"""Entrena recomendador Item-Item CF filtrando el top 5 de productos por cliente.

Mismo enfoque que ``train_recommender.py`` pero en lugar de filtrar
por frecuencia global, selecciona los 5 productos más comprados
de cada cliente. Las recomendaciones se guardan como lista de nombres.

Uso::

    python src/train_recommender_by_client.py --config params.yaml
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.sparse import save_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collaborative import (
    compute_item_similarity,
    create_sparse_matrix,
    recommend_for_client,
)
from config import load_config, processed_path
from data_io import load_parquet, load_product_catalog, save_parquet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_and_save(config_path: str | Path) -> None:
    """Pipeline: top 5 productos por cliente, Item-Item CF, guarda como lista de nombres."""
    start_time = time.time()
    logger.info("--- Recomendador Item-Item (top 5 productos por cliente) ---")

    cfg = load_config(config_path)
    rec_cfg = cfg["recommendations_item_item"]
    data_cfg = cfg["data"]
    proc = processed_path(cfg)

    model_out = Path(rec_cfg["model_folder"])
    output_dir = proc / rec_cfg["output_folder"]
    model_out.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_recs = rec_cfg["num_recommendations"]

    # Cargar transacciones
    transactions = load_parquet(
        proc / rec_cfg["transaction_data_processed_file"], "Transacciones"
    )
    transactions = (
        transactions.rename(columns={"id_client": "client", "date_sale": "date"})
        .dropna(subset=["client", "product", "date"])
        .assign(
            client=lambda d: d["client"].astype(str),
            product=lambda d: d["product"].astype(str).str.strip(),
        )
    )

    # Ventana temporal
    max_date = transactions["date"].max()
    cutoff = max_date - relativedelta(months=data_cfg.get("last_month_with_sale", 3))
    tx_recent = transactions[transactions["date"] >= cutoff].copy()
    logger.info(
        "Ventana: %s -> %s (%d filas)", cutoff.date(), max_date.date(), len(tx_recent)
    )

    if tx_recent.empty:
        logger.error("Sin transacciones en la ventana temporal.")
        return

    # Top 5 productos por cliente
    logger.info("Seleccionando top 5 productos por cliente...")
    client_prod_counts = (
        tx_recent.groupby(["client", "product"])
        .size()
        .reset_index(name="count")
        .sort_values(["client", "count"], ascending=[True, False])
    )
    top5 = client_prod_counts.groupby("client").head(5).reset_index(drop=True)
    tx_filtered = (
        tx_recent.merge(
            top5[["client", "product"]], on=["client", "product"], how="inner"
        )
        .drop_duplicates()
        .copy()
    )
    logger.info("Interacciones tras filtro top 5: %d", len(tx_filtered))

    if tx_filtered.empty:
        logger.error("DataFrame filtrado vacío.")
        return

    # Matriz dispersa + similitud
    matrix, mappings = create_sparse_matrix(tx_filtered)
    if matrix is None:
        return
    similarity = compute_item_similarity(matrix)
    if similarity is None:
        return

    # Mapeo producto ID → nombre
    catalog = load_product_catalog(proc / data_cfg["products_file"])
    name_map = catalog.set_index("product")["description"].to_dict()

    # Precalcular recomendaciones como lista de nombres
    total = len(mappings["user_map_inv"])
    logger.info("Generando %d recs para %d clientes...", n_recs, total)

    rows = []
    for client_str, client_idx in mappings["user_map_inv"].items():
        rec_ids = recommend_for_client(
            client_idx,
            matrix,
            similarity,
            mappings["product_map"],
            n_recs,
            return_scores=False,
        )
        names = [name_map.get(pid, pid) for pid in rec_ids]
        rows.append({"client": client_str, "recommended_products": names})

    recs_df = pd.DataFrame(rows)

    # Guardar artefactos
    save_npz(model_out / rec_cfg["similarity_matrix_file"], similarity)
    save_npz(model_out / rec_cfg["user_item_matrix_file"], matrix)
    with open(model_out / rec_cfg["mappings_file"], "wb") as f:
        pickle.dump(mappings, f)

    if not recs_df.empty:
        save_parquet(
            recs_df,
            output_dir / rec_cfg["precomputed_recs_file"],
            "Recomendaciones por cliente",
        )

    logger.info("Finalizado en %.2f segundos", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena recomendador Item-Item CF (top 5 por cliente)."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    train_and_save(args.config)
