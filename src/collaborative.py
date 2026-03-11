"""Filtrado colaborativo Item-Item.

Módulo compartido entre ``train_recommender.py`` (top 10% global)
y ``train_recommender_by_client.py`` (top 5 por cliente).

Funcionalidad:
    - Construcción de matriz dispersa usuario-item.
    - Cálculo de similitud coseno item-item.
    - Generación de recomendaciones por cliente.
"""

import logging
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def create_sparse_matrix(
    df: pd.DataFrame,
    client_col: str = "client",
    product_col: str = "product",
) -> tuple[csr_matrix | None, dict[str, Any]]:
    """Crea la matriz dispersa usuario-item y mapeos de índices.

    Args:
        df: DataFrame con al menos columnas ``client_col`` y ``product_col``.
        client_col: Nombre de la columna de cliente.
        product_col: Nombre de la columna de producto.

    Returns:
        ``(sparse_matrix, mappings)`` donde ``mappings`` contiene:
            - ``user_map``: idx -> client_id
            - ``product_map``: idx -> product_id
            - ``user_map_inv``: client_id -> idx
        Si no hay datos suficientes, retorna ``(None, {})``.
    """
    counts = df.groupby([client_col, product_col]).size().reset_index(name="count")

    # Codificar categorías UNA sola vez para evitar desalineación de índices
    client_cat = counts[client_col].astype("category")
    product_cat = counts[product_col].astype("category")

    counts = counts.assign(
        client_id=client_cat.cat.codes,
        product_id=product_cat.cat.codes,
    )

    user_map = dict(enumerate(client_cat.cat.categories))
    product_map = dict(enumerate(product_cat.cat.categories))
    user_map_inv = {v: k for k, v in user_map.items()}

    matrix = csr_matrix(
        (counts["count"], (counts["client_id"], counts["product_id"])),
        shape=(len(user_map), len(product_map)),
    )

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        logger.warning("Matriz vacía — datos insuficientes.")
        return None, {}

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    logger.info(
        "Matriz %d usuarios x %d productos | %d interacciones | densidad: %.4f",
        len(user_map),
        len(product_map),
        len(counts),
        density,
    )

    mappings = {
        "user_map": user_map,
        "product_map": product_map,
        "user_map_inv": user_map_inv,
    }
    return matrix, mappings


def compute_item_similarity(matrix: csr_matrix) -> csr_matrix | None:
    """Calcula la matriz de similitud coseno item-item.

    Returns:
        Matriz dispersa de similitud, o ``None`` si hay menos de 2 productos.
    """
    if matrix is None or matrix.shape[1] < 2:
        logger.warning("No se puede calcular similitud (< 2 productos).")
        return None
    logger.info("Calculando similitud Item-Item (coseno)...")
    similarity = cosine_similarity(matrix.T, dense_output=False)
    logger.info("Similitud calculada: %s", similarity.shape)
    return similarity


def recommend_for_client(
    client_idx: int,
    sparse_matrix: csr_matrix,
    similarity_matrix: csr_matrix,
    product_map: dict[int, str],
    n_recs: int,
    return_scores: bool = False,
) -> list[dict[str, Any]] | list[str]:
    """Genera recomendaciones item-item para un cliente.

    Para cada producto comprado, pondera la similitud con productos no comprados
    por la frecuencia de compra, y ordena de mayor a menor score.

    Args:
        client_idx: Índice del cliente en la matriz dispersa.
        sparse_matrix: Matriz usuario-item.
        similarity_matrix: Matriz de similitud item-item.
        product_map: Mapeo ``idx -> product_id_str``.
        n_recs: Número máximo de recomendaciones.
        return_scores: Si ``True``, retorna dicts con ``recommended_product`` y ``score``.
    """
    if client_idx >= sparse_matrix.shape[0]:
        return []

    purchases = sparse_matrix[client_idx, :]
    purchased_indices = purchases.indices
    purchase_values = purchases.data

    if len(purchased_indices) == 0:
        return []

    purchased_set = set(purchased_indices)
    scores: dict[int, float] = {}

    for i, item_idx in enumerate(purchased_indices):
        if item_idx >= similarity_matrix.shape[0]:
            continue
        sim_row = similarity_matrix[item_idx].toarray().flatten()
        for j, sim_score in enumerate(sim_row):
            if j not in purchased_set and sim_score > 0:
                scores[j] = scores.get(j, 0) + sim_score * purchase_values[i]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]

    if return_scores:
        return [
            {"recommended_product": product_map[idx], "recommendation_score": score}
            for idx, score in ranked
            if idx in product_map
        ]
    return [product_map[idx] for idx, _ in ranked if idx in product_map]
