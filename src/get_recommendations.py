#!/usr/bin/env python3
"""Une recomendaciones precalculadas con predicciones de compra.

Soporta tanto recomendaciones item-item como por clustering.
Busca los archivos de recomendaciones precalculadas y los une
con el archivo de predicciones indicado.

Uso::

    # Item-item
    python src/get_recommendations.py \\
        --input_file predictions/predictions_client.csv \\
        --output_file predictions/predictions_with_recs.csv

    # Clustering + item-item combinados
    python src/get_recommendations.py \\
        --input_file predictions/predictions_client.csv \\
        --output_file predictions/predictions_with_all_recs.csv \\
        --include_clustering
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, processed_path
from data_io import load_product_catalog

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _load_recs_file(path: Path, label: str) -> pd.DataFrame | None:
    """Intenta cargar un archivo de recomendaciones Parquet."""
    if not path.exists():
        logger.warning("%s no encontrado: %s", label, path)
        return None
    df = pd.read_parquet(path)
    df["client"] = df["client"].astype(str)
    logger.info(
        "%s cargado: %d filas, %d clientes", label, len(df), df["client"].nunique()
    )
    return df


def _merge_recommendations(
    predictions: pd.DataFrame,
    recs: pd.DataFrame | None,
    rec_col_name: str,
) -> pd.DataFrame:
    """Une recomendaciones con predicciones por cliente."""
    if recs is None:
        return predictions

    # Determinar columna de productos recomendados
    rec_cols = ["client"]
    if "recommended_products" in recs.columns:
        rec_cols.append("recommended_products")
    elif "recommended_product" in recs.columns:
        rec_cols.append("recommended_product")
        if "recommendation_score" in recs.columns:
            rec_cols.append("recommendation_score")
        if "recommendation_rank" in recs.columns:
            rec_cols.append("recommendation_rank")

    result = predictions.merge(recs[rec_cols], on="client", how="left")

    # Renombrar para distinguir entre tipos de recomendación
    if (
        "recommended_products" in result.columns
        and rec_col_name != "recommended_products"
    ):
        result.rename(columns={"recommended_products": rec_col_name}, inplace=True)

    # Verificar cuántos clientes recibieron recomendaciones
    rec_check_cols = [c for c in result.columns if c not in predictions.columns]
    if rec_check_cols and len(result) > 0:
        found = result[result[rec_check_cols[0]].notna()]["client"].nunique()
    else:
        found = 0
    logger.info("Recomendaciones %s encontradas para %d clientes", rec_col_name, found)
    return result


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def get_recommendations(
    input_file: str,
    output_file: str,
    config_path: str = "params.yaml",
    include_clustering: bool = False,
) -> None:
    """Carga predicciones y les une recomendaciones precalculadas."""
    cfg = load_config(config_path)
    rec_cfg = cfg["recommendations_item_item"]
    proc = processed_path(cfg)

    # Cargar predicciones
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("Archivo de entrada no encontrado: %s", input_path)
        sys.exit(1)

    predictions = pd.read_csv(input_path)
    if "client" not in predictions.columns:
        logger.error("El archivo de entrada debe contener columna 'client'.")
        sys.exit(1)
    predictions["client"] = predictions["client"].astype(str)

    # Recomendaciones item-item
    item_item_path = proc / rec_cfg["output_folder"] / rec_cfg["precomputed_recs_file"]
    item_item_recs = _load_recs_file(item_item_path, "Item-item recs")
    result = _merge_recommendations(
        predictions, item_item_recs, "item_item_recommended_products"
    )

    # Recomendaciones por clustering (opcional)
    if include_clustering:
        cluster_cfg = cfg.get("recommendations_clustering", {})
        cluster_path = (
            proc
            / cluster_cfg.get("output_folder", "recommendations")
            / cluster_cfg.get(
                "precomputed_cluster_recs_file", "precomputed_cluster_recs.parquet"
            )
        )
        cluster_recs = _load_recs_file(cluster_path, "Clustering recs")
        result = _merge_recommendations(
            result, cluster_recs, "cluster_recommended_products"
        )

    # Agregar detalles de producto (si hay columna recommended_product)
    if "recommended_product" in result.columns:
        try:
            catalog = load_product_catalog(proc / cfg["data"]["products_file"])
            catalog.rename(columns={"product": "recommended_product"}, inplace=True)
            result = result.merge(catalog, on="recommended_product", how="left")
        except FileNotFoundError:
            logger.warning("Catálogo no encontrado. Sin detalles de producto.")

    # Guardar
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sort_cols = [c for c in predictions.columns if c in result.columns]
    if "recommendation_rank" in result.columns:
        sort_cols.append("recommendation_rank")
    result.sort_values(sort_cols, na_position="last", inplace=True)

    result.to_csv(out_path, index=False, float_format="%.4f")
    logger.info("Resultados guardados: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Une recomendaciones precalculadas con predicciones."
    )
    parser.add_argument(
        "--input_file", required=True, help="CSV de entrada con columna 'client'."
    )
    parser.add_argument("--output_file", required=True, help="CSV de salida.")
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml.")
    parser.add_argument(
        "--include_clustering", action="store_true", help="Incluir recs por clustering."
    )
    args = parser.parse_args()
    get_recommendations(
        args.input_file, args.output_file, args.config, args.include_clustering
    )
