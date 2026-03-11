"""Análisis de patrones de compra de clientes.

Funcionalidad compartida entre ``preprocess.py`` (modelo predictivo)
y ``train_recommender_by_clustering.py`` (recomendación por segmentos).
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_recent_clients(
    df: pd.DataFrame,
    months: int,
    date_col: str = "date_sale",
    client_col: str = "id_client",
) -> pd.DataFrame:
    """Mantiene solo clientes con compras en los últimos ``months`` meses.

    Usa la fecha máxima del dataset como referencia (no ``datetime.now()``),
    para que funcione correctamente con datos históricos.

    Returns:
        DataFrame filtrado. Puede estar vacío si no hay clientes recientes.
    """
    max_date = df[date_col].max()
    cutoff = max_date - timedelta(days=months * 30)
    recent_ids = df[df[date_col] >= cutoff][client_col].unique()
    logger.info(
        "Clientes últimos %d meses (ref: %s, corte: %s): %d",
        months,
        max_date.date(),
        cutoff.date(),
        len(recent_ids),
    )
    if len(recent_ids) == 0:
        logger.warning("No se encontraron clientes con actividad reciente.")
    return df[df[client_col].isin(recent_ids)].copy()


def compute_purchase_patterns(
    df: pd.DataFrame,
    client_col: str = "id_client",
    date_col: str = "date_sale",
    product_col: str = "product",
    amount_col: str | None = None,
) -> pd.DataFrame:
    """Calcula métricas de patrón de compra por cliente.

    Columnas de salida:
        - ``client_col``: identificador del cliente.
        - ``purchase_days``: días distintos con compras.
        - ``median_days_between``: mediana de días entre compras.
        - ``std_days_between``: desviación estándar de días entre compras.
        - ``product_distinct``: número de productos únicos comprados.
        - ``pay_amount_mean`` (si se pasa ``amount_col``): monto promedio por día de compra.

    Note:
        El primer día de compra de cada cliente produce NaN en ``days_between``,
        que es excluido automáticamente por ``median()`` y ``std()``.
    """
    # Productos únicos por cliente
    product_counts = (
        df.groupby(client_col)[product_col].nunique().rename("product_distinct")
    )

    # Métricas temporales sobre fechas únicas
    daily = (
        df.drop_duplicates(subset=[client_col, date_col])[[client_col, date_col]]
        .sort_values([client_col, date_col])
        .copy()
    )
    daily["days_between"] = daily.groupby(client_col)[date_col].diff().dt.days

    patterns = (
        daily.groupby(client_col)
        .agg(
            purchase_days=(date_col, "nunique"),
            median_days_between=("days_between", "median"),
            std_days_between=("days_between", "std"),
        )
        .reset_index()
    )
    patterns = patterns.merge(product_counts, on=client_col, how="left")

    # Monto promedio (usado por clustering)
    if amount_col and amount_col in df.columns:
        amount_mean = (
            df.drop_duplicates(subset=[client_col, date_col])
            .groupby(client_col)[amount_col]
            .mean()
            .rename("pay_amount_mean")
        )
        patterns = patterns.merge(amount_mean, on=client_col, how="left")

    return patterns


def apply_pattern_filters(
    patterns: pd.DataFrame,
    min_purchase_days: int,
    max_median_days: float,
    max_std_days: float,
    min_products: int | None = None,
    client_col: str = "id_client",
) -> pd.Series:
    """Filtra clientes por criterios de patrón de compra.

    Returns:
        Serie con los IDs de clientes que cumplen todos los criterios.
    """
    mask = (
        (patterns["purchase_days"] >= min_purchase_days)
        & (patterns["median_days_between"] < max_median_days)
        & (patterns["std_days_between"] <= max_std_days)
    )
    if min_products is not None:
        mask = mask & (patterns["product_distinct"] >= min_products)

    filtered = patterns[mask]
    logger.info(
        "Filtro de patrones: %d -> %d clientes válidos",
        len(patterns),
        len(filtered),
    )
    return filtered[client_col]


# ---------------------------------------------------------------------------
#  Features para segmentación de clientes
# ---------------------------------------------------------------------------


def compute_segmentation_features(
    df: pd.DataFrame,
    client_col: str = "id_client",
    date_col: str = "date_sale",
    product_col: str = "product",
    amount_col: str = "invoice_value_with_discount_and_without_iva",
) -> pd.DataFrame:
    """Calcula features orientadas a identificar perfiles de negocio.

    Diseñadas para separar 5 tipos de cliente:

    - **Ballena**: muchos tickets/mes, alto valor (mini-mercados).
    - **Cotidiano**: 1-3 tickets/mes, ticket medio (familia que merca).
    - **Mensual**: ~1 ticket/mes, compra puntual.
    - **Hormiga**: muchas visitas, bajo monto por ticket.
    - **Esporádico**: pocos meses activos, irregular, no predecible.

    Features calculadas:
        - ``tickets_per_month``: promedio de tickets (días con compra) por mes.
        - ``ticket_median``: valor mediano del ticket en COP.
        - ``months_active_ratio``: fracción de meses con al menos 1 compra.
        - ``products_per_visit``: productos distintos promedio por visita.
        - ``product_distinct``: total de productos únicos comprados.
        - ``monetary_total``: gasto acumulado total en COP.

    Returns:
        DataFrame con una fila por cliente y todas las features.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Meses totales en la ventana de observación
    total_months = max(
        1,
        (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1,
    )

    # --- Tickets únicos por (cliente, fecha) ---
    ticket_daily = df.drop_duplicates(subset=[client_col, date_col])

    # --- Tickets por mes ---
    total_tickets = (
        ticket_daily.groupby(client_col)[date_col].nunique().rename("_total_tickets")
    )

    # --- Meses activos ---
    ticket_daily_m = ticket_daily[[client_col, date_col]].copy()
    ticket_daily_m["_month"] = ticket_daily_m[date_col].dt.to_period("M")
    months_active = (
        ticket_daily_m.groupby(client_col)["_month"].nunique().rename("_months_active")
    )

    freq = total_tickets.to_frame().merge(months_active, on=client_col, how="left")
    freq["_months_active"] = freq["_months_active"].clip(
        lower=1
    )  # Evitar división por cero
    freq["tickets_per_month"] = freq["_total_tickets"] / freq["_months_active"]
    freq["months_active_ratio"] = freq["_months_active"] / total_months

    # --- Valor del ticket (suma de líneas por día de compra) ---
    if amount_col in df.columns:
        daily_amount = (
            df.groupby([client_col, date_col])[amount_col].sum().reset_index()
        )
        monetary = (
            daily_amount.groupby(client_col)[amount_col]
            .agg(ticket_median="median", monetary_total="sum")
            .reset_index()
        )
    else:
        monetary = pd.DataFrame({client_col: df[client_col].unique()})
        monetary["ticket_median"] = np.nan
        monetary["monetary_total"] = np.nan

    # --- Productos por visita ---
    products_per_day = (
        df.groupby([client_col, date_col])[product_col]
        .nunique()
        .reset_index()
        .rename(columns={product_col: "n_products"})
    )
    basket = (
        products_per_day.groupby(client_col)
        .agg(products_per_visit=("n_products", "mean"))
        .reset_index()
    )

    # Productos únicos totales
    product_distinct = (
        df.groupby(client_col)[product_col].nunique().rename("product_distinct")
    )

    # --- Merge final ---
    features = freq[["tickets_per_month", "months_active_ratio"]].reset_index()
    features = features.merge(monetary, on=client_col, how="left")
    features = features.merge(basket, on=client_col, how="left")
    features = features.merge(product_distinct, on=client_col, how="left")

    logger.info(
        "Features de segmentación: %d clientes, ventana=%d meses, features=%d",
        len(features),
        total_months,
        len(features.columns) - 1,
    )
    return features


# Etiquetas de segmento según centroides
SEGMENT_LABELS = {
    "ballena": "Ballena (mini-mercado)",
    "cotidiano": "Cotidiano (familia)",
    "mensual": "Mensual",
    "hormiga": "Hormiga (frecuente, bajo monto)",
    "esporadico": "Esporádico",
}


def label_clusters(
    features: pd.DataFrame,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Asigna etiquetas de negocio a cada cluster basándose en los centroides.

    Reglas (aplicadas sobre la media del cluster):
        1. Esporádico: ``months_active_ratio`` más bajo.
        2. Ballena: ``ticket_median`` más alto (excluyendo esporádico).
        3. Hormiga: ``tickets_per_month`` más alto Y ``ticket_median`` más bajo
           (excluyendo esporádico y ballena).
        4. Cotidiano / Mensual: se distinguen por ``tickets_per_month``
           (cotidiano > mensual).

    Returns:
        DataFrame con columna ``segment`` añadida.
    """
    means = features.groupby(cluster_col)[
        ["tickets_per_month", "ticket_median", "months_active_ratio"]
    ].mean()

    labels = {}
    remaining = set(means.index)

    # 1. Esporádico: menor months_active_ratio
    esporadico_id = means.loc[list(remaining), "months_active_ratio"].idxmin()
    labels[esporadico_id] = "esporadico"
    remaining.discard(esporadico_id)

    # 2. Ballena: mayor ticket_median (de los restantes)
    ballena_id = means.loc[list(remaining), "ticket_median"].idxmax()
    labels[ballena_id] = "ballena"
    remaining.discard(ballena_id)

    # 3. Hormiga: mayor tickets_per_month Y menor ticket_median de los restantes
    if len(remaining) >= 2:
        rest_means = means.loc[list(remaining)]
        # Hormiga = quien tiene la combinación de más frecuencia y menor ticket
        # Usamos ratio: tickets_per_month / ticket_median (normalizado)
        normed = rest_means.copy()
        for col in normed.columns:
            r = normed[col].max() - normed[col].min()
            normed[col] = (normed[col] - normed[col].min()) / r if r > 0 else 0.5
        score_hormiga = normed["tickets_per_month"] - normed["ticket_median"]
        hormiga_id = score_hormiga.idxmax()
        labels[hormiga_id] = "hormiga"
        remaining.discard(hormiga_id)

    # 4. Cotidiano y Mensual por tickets_per_month
    if len(remaining) >= 2:
        rest_sorted = means.loc[list(remaining), "tickets_per_month"].sort_values()
        labels[rest_sorted.index[0]] = "mensual"
        labels[rest_sorted.index[-1]] = "cotidiano"
        remaining -= set(rest_sorted.index[:2])
    elif len(remaining) == 1:
        labels[remaining.pop()] = "cotidiano"

    # Etiquetar cualquier cluster sobrante
    for cid in remaining:
        labels[cid] = "cotidiano"

    features = features.copy()
    features["segment"] = features[cluster_col].map(labels)
    features["segment_label"] = features["segment"].map(SEGMENT_LABELS)

    summary = (
        features.groupby("segment")
        .agg(
            clientes=(cluster_col, "count"),
            tickets_mes=("tickets_per_month", "mean"),
            ticket_mediano=("ticket_median", "mean"),
            meses_activos=("months_active_ratio", "mean"),
        )
        .round(1)
    )
    logger.info("Segmentos asignados:\n%s", summary.to_string())

    return features
