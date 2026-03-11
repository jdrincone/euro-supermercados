#!/usr/bin/env python3
"""Entrena recomendador Item-Item CF filtrando el top 10% de productos globales.

Flujo:
    1. Carga transacciones procesadas y catálogo de productos.
    2. Filtra el 10% de productos más frecuentes globalmente.
    3. Construye matriz dispersa usuario-item.
    4. Calcula similitud coseno item-item.
    5. Precalcula recomendaciones para todos los clientes.
    6. Guarda artefactos (matrices, mapeos, recomendaciones).

Uso::

    python src/train_recommender.py --config params.yaml
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
    """Pipeline completo: filtra top 10% productos, entrena CF y guarda artefactos."""
    start_time = time.time()
    logger.info("--- Recomendador Item-Item (top 10%% productos globales) ---")

    cfg = load_config(config_path)
    rec_cfg = cfg["recommendations_item_item"]
    data_cfg = cfg["data"]
    proc = processed_path(cfg)

    # Rutas
    model_out = Path(rec_cfg["model_folder"])
    output_dir = proc / rec_cfg["output_folder"]
    model_out.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_recs = rec_cfg["num_recommendations"]

    # Cargar datos
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

    catalog = load_product_catalog(proc / data_cfg["products_file"])

    # Filtrar productos excluidos
    excluded = rec_cfg.get(
        "excluded_product_descriptions",
        cfg.get("recommendations_clustering", {}).get(
            "excluded_product_descriptions", []
        ),
    )
    tx_with_desc = transactions.merge(catalog, on="product", how="left")
    tx_filtered = tx_with_desc[
        (~tx_with_desc["description"].isin(excluded))
        & (tx_with_desc["description"].notna())
    ].copy()

    if tx_filtered.empty:
        logger.error("Sin transacciones tras excluir productos.")
        return

    # Top 10% productos por frecuencia global
    freq = tx_filtered["description"].value_counts()
    q90 = freq.quantile(0.90)
    top_desc = freq[freq >= q90].index
    top_ids = catalog[catalog["description"].isin(top_desc)]["product"].unique()
    tx_filtered = tx_filtered[tx_filtered["product"].isin(top_ids)].copy()
    logger.info(
        "Productos top 10%%: %d descripciones, %d IDs", len(top_desc), len(top_ids)
    )

    if tx_filtered.empty:
        logger.error("Sin transacciones tras filtro de top 10%%.")
        return

    # Ventana temporal
    max_date = tx_filtered["date"].max()
    cutoff = max_date - relativedelta(months=data_cfg.get("last_month_with_sale", 3))
    tx_final = tx_filtered[tx_filtered["date"] >= cutoff].copy()
    logger.info(
        "Ventana: %s -> %s (%d filas)", cutoff.date(), max_date.date(), len(tx_final)
    )

    # Matriz dispersa + similitud
    matrix, mappings = create_sparse_matrix(tx_final)
    if matrix is None:
        return
    similarity = compute_item_similarity(matrix)
    if similarity is None:
        return

    # Precalcular recomendaciones
    total = len(mappings["user_map_inv"])
    logger.info("Generando %d recs para %d clientes...", n_recs, total)

    all_recs = []
    for i, (client_str, client_idx) in enumerate(mappings["user_map_inv"].items()):
        recs = recommend_for_client(
            client_idx,
            matrix,
            similarity,
            mappings["product_map"],
            n_recs,
            return_scores=True,
        )
        for rec in recs:
            rec["client"] = client_str
        all_recs.extend(recs)
        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info("  %d/%d clientes", i + 1, total)

    recs_df = pd.DataFrame(all_recs)
    if not recs_df.empty:
        recs_df["recommendation_rank"] = (
            recs_df.groupby("client")["recommendation_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        recs_df = recs_df.sort_values(["client", "recommendation_rank"])
    else:
        logger.warning("No se generaron recomendaciones.")

    # Guardar artefactos
    save_npz(model_out / rec_cfg["similarity_matrix_file"], similarity)
    save_npz(model_out / rec_cfg["user_item_matrix_file"], matrix)
    with open(model_out / rec_cfg["mappings_file"], "wb") as f:
        pickle.dump(mappings, f)

    if not recs_df.empty:
        save_parquet(
            recs_df,
            output_dir / rec_cfg["precomputed_recs_file"],
            "Recomendaciones item-item",
        )

    logger.info("Finalizado en %.2f segundos", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena recomendador Item-Item CF (top 10%% productos)."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    train_and_save(args.config)
