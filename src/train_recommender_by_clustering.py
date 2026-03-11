#!/usr/bin/env python3
"""Entrena recomendador basado en segmentación de clientes (K-Means).

Segmenta clientes en 5 perfiles de negocio y recomienda los productos
más populares dentro de cada segmento:

- **Ballena**: muchos tickets/mes, alto valor (mini-mercados).
- **Cotidiano**: 1-3 tickets/mes, ticket medio (familia que merca).
- **Mensual**: ~1 ticket/mes, compra puntual.
- **Hormiga**: muchas visitas, bajo monto por ticket.
- **Esporádico**: pocos meses activos, irregular.

Uso::

    python src/train_recommender_by_clustering.py --config params.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client_filters import validate_client_ids
from config import load_config, processed_path
from data_io import load_product_catalog, save_parquet
from patterns import (
    compute_segmentation_features,
    filter_recent_clients,
    label_clusters,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Carga de ventas
# ---------------------------------------------------------------------------


def _load_sales(path: Path) -> pd.DataFrame:
    """Carga ventas con validación robusta de IDs y fechas."""
    df = pd.read_parquet(path)
    df = df.assign(id_client=df["id_client"].astype(str).str.strip())
    df = validate_client_ids(df, id_col="id_client")
    df = df.assign(
        product=df["product"].astype(str).str.strip(),
        date_sale=pd.to_datetime(df["date_sale"]).dt.normalize(),
    )
    df = df.dropna(subset=["date_sale"])
    logger.info(
        "Ventas cargadas: %d filas, %d clientes", len(df), df["id_client"].nunique()
    )
    return df


# ---------------------------------------------------------------------------
#  Clustering
# ---------------------------------------------------------------------------

# Features con distribución sesgada que requieren log-transform
_LOG_FEATURES = {"ticket_median", "monetary_total"}


def _prepare_features(features_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Prepara la matriz de features: imputa, log-transforma y escala."""
    X = features_df[feature_cols].copy()
    X = X.fillna(X.median())
    for col in feature_cols:
        if col in _LOG_FEATURES:
            X[col] = np.log1p(X[col].clip(lower=0))
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def _find_optimal_k(X_scaled: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    """Encuentra el k óptimo por silhouette score."""
    k_max = min(k_max, len(X_scaled) - 1)
    if k_min >= k_max:
        return k_min

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        logger.info("  k=%d → silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("K óptimo: %d (silhouette=%.4f)", best_k, best_score)
    return best_k


def _perform_clustering(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int | None,
    seed: int,
    k_min: int = 4,
    k_max: int = 7,
) -> pd.DataFrame:
    """Ejecuta K-Means y asigna etiquetas de segmento."""
    if features_df.empty:
        logger.warning("Sin clientes para clusterizar.")
        return features_df

    X_scaled = _prepare_features(features_df, feature_cols)

    if n_clusters is None or n_clusters <= 0:
        logger.info("Buscando k óptimo (rango %d-%d)...", k_min, k_max)
        n_clusters = _find_optimal_k(X_scaled, k_min, k_max, seed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    features_df = features_df.copy()
    features_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Asignar etiquetas de negocio automáticamente
    features_df = label_clusters(features_df)

    return features_df


def _generate_cluster_recommendations(
    sales: pd.DataFrame,
    features_df: pd.DataFrame,
    catalog: pd.DataFrame,
    excluded: list[str],
) -> pd.DataFrame:
    """Recomienda los productos más populares dentro de cada segmento."""
    df = sales.merge(
        features_df[["id_client", "cluster", "segment"]],
        on="id_client",
        how="inner",
    )
    df = df.merge(catalog, on="product", how="left")
    df = df[~df["description"].isin(excluded)].dropna(subset=["description"])

    cluster_avg = features_df.groupby("cluster")["product_distinct"].mean()

    cluster_recs: dict[int, list[str]] = {}
    for cid in sorted(features_df["cluster"].unique()):
        top_n = max(1, int(round(cluster_avg.get(cid, 5))))
        cluster_sales = df[df["cluster"] == cid]
        top_products = (
            cluster_sales["description"].value_counts().nlargest(top_n).index.tolist()
            if not cluster_sales.empty
            else []
        )
        cluster_recs[cid] = top_products
        seg = features_df.loc[features_df["cluster"] == cid, "segment"].iloc[0]
        logger.info("Cluster %d (%s): %d productos", cid, seg, len(top_products))

    result = features_df[["id_client", "cluster", "segment", "segment_label"]].copy()
    result["recommended_products"] = result["cluster"].map(cluster_recs)
    result = result.rename(
        columns={"id_client": "client", "cluster": "purchase_pattern_cluster"}
    )
    return result


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------


def main_pipeline(config_path: str | Path) -> None:
    """Orquesta segmentación + recomendación por segmentos."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    rec_cfg = cfg["recommendations_clustering"]
    seed = cfg["base"]["random_state"]
    proc = processed_path(cfg)

    output_dir = proc / rec_cfg["output_folder"]
    output_file = output_dir / rec_cfg["precomputed_cluster_recs_file"]
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_cols = [
        "client",
        "purchase_pattern_cluster",
        "segment",
        "segment_label",
        "recommended_products",
    ]
    empty_df = pd.DataFrame(columns=empty_cols)

    # Cargar datos
    df_sales = _load_sales(proc / data_cfg["sales_file"])
    catalog = load_product_catalog(proc / data_cfg["products_file"])

    # Filtrar clientes recientes
    df_recent = filter_recent_clients(df_sales, rec_cfg["months_recent_activity"])
    if df_recent.empty:
        logger.warning("Sin clientes recientes. Guardando archivo vacío.")
        save_parquet(empty_df, output_file, "Recs vacías")
        return

    # Features de segmentación (para TODOS los clientes, no filtramos por patrón)
    features = compute_segmentation_features(
        df_recent,
        amount_col="invoice_value_with_discount_and_without_iva",
    )

    if features.empty:
        logger.warning("Sin clientes válidos. Guardando archivo vacío.")
        save_parquet(empty_df, output_file, "Recs vacías")
        return

    # Clustering con etiquetas de segmento
    feature_cols = rec_cfg["features_for_clustering"]
    n_clusters = rec_cfg.get("n_clusters")

    features = _perform_clustering(
        features,
        feature_cols,
        n_clusters,
        seed,
        k_min=rec_cfg.get("k_min", 4),
        k_max=rec_cfg.get("k_max", 7),
    )

    # Guardar asignaciones completas
    assignments_path = proc / "cluster_assignments.parquet"
    save_parquet(features, assignments_path, "Asignaciones de segmento")

    # Recomendaciones por segmento
    recs = _generate_cluster_recommendations(
        df_recent,
        features,
        catalog,
        rec_cfg.get("excluded_product_descriptions", []),
    )

    save_parquet(recs, output_file, "Recomendaciones por segmento")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentación de clientes + recomendaciones."
    )
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    args = parser.parse_args()
    main_pipeline(args.config)
