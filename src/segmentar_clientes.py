#!/usr/bin/env python3
"""Segmentación de clientes VIP de Euro Supermercados.

Lee df_vip (parquet o CSV), aplica K-Means (k=4) y exporta un parquet
con cluster + datos de contacto de la API.

Uso:
    uv run python src/segmentar_clientes.py

    # Sin consultar API de terceros (solo cluster + punto de venta)
    uv run python src/segmentar_clientes.py --sin-contacto

    # Cambiar archivo de salida
    uv run python src/segmentar_clientes.py --output data/processed/mis_clusters.parquet

Salida:
    data/processed/clientes_segmentados.parquet

    Columnas: user_id, cluster_id, cluster_descripcion, id_point_sale,
              name, email, phone, document_type, country, department, town, gender
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
N_CLUSTERS = 4
FEATURE_COLS = [
    "ticket_promedio", "valor_tipico", "gasto_total_vida",
    "volatilidad_gasto", "frecuencia_total", "dias_sin_comprar",
]

# IDs genéricos que no corresponden a personas reales
_BLACKLIST_IDS = frozenset({
    "0", "1", "12345", "123456", "1234567", "12345678",
    "123456789", "1234567890",
    *(f"{d}" * n for d in range(10) for n in range(6, 11)),
})

# Nombres de los 4 segmentos
_SEGMENT_DESCRIPTIONS = {
    "reposicion": "Clientes de Reposición — compras frecuentes, ticket bajo (~$50k)",
    "intermedio": "Mercado Intermedio — frecuencia media, riesgo de pérdida (~$84k)",
    "masa_critica": "Masa Crítica — mayor volumen de clientes, foco de crecimiento (~$93k)",
    "ballenas": "Ballenas (Minisupermercados) — ticket alto, alto valor individual (~$390k)",
}


# ---------------------------------------------------------------------------
#  1. Carga
# ---------------------------------------------------------------------------


def cargar_datos_vip() -> pd.DataFrame:
    """Carga df_vip desde parquet o CSV."""
    parquet = PROCESSED_DIR / "df_vip.parquet"
    csv = PROCESSED_DIR / "df_vip.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        logger.info("Cargado %s: %d filas", parquet.name, len(df))
    elif csv.exists():
        df = pd.read_csv(csv, low_memory=False)
        logger.info("Cargado %s (CSV legacy): %d filas", csv.name, len(df))
    else:
        raise FileNotFoundError(
            f"No se encontró df_vip en {PROCESSED_DIR}. "
            "Ejecuta: uv run python src/download_vip.py --download --process"
        )

    df["date_sale"] = pd.to_datetime(df["date_sale"])
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df[df["tiket_price"] > 0].copy()
    return df


# ---------------------------------------------------------------------------
#  2. Limpieza de IDs
# ---------------------------------------------------------------------------


def limpiar_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra user_ids inválidos (cédula colombiana 6-10 dígitos)."""
    n_antes = len(df)
    ids = df["user_id"]

    mask = (
        ids.str.fullmatch(r"\d+", na=False)
        & ~ids.str.startswith("0", na=False)
        & ids.str.len().between(6, 10)
        & ~ids.isin(_BLACKLIST_IDS)
        & ids.apply(lambda s: len(set(s)) > 1)
    )

    df = df[mask].copy()
    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        logger.info("IDs limpiados: %d filas eliminadas (%.2f%%)", n_eliminados, n_eliminados / n_antes * 100)
    return df


# ---------------------------------------------------------------------------
#  3. Features + Clustering
# ---------------------------------------------------------------------------


def calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula 6 features por cliente a partir de tickets únicos."""
    df_tickets = df[["tiket_id", "user_id", "tiket_price", "date_sale"]].drop_duplicates()
    ref_date = df_tickets["date_sale"].max()

    features = df_tickets.groupby("user_id").agg({
        "tiket_price": ["mean", "median", "sum", "std"],
        "tiket_id": "count",
        "date_sale": lambda x: (ref_date - x.max()).days,
    }).fillna(0)

    features.columns = FEATURE_COLS
    logger.info("Features: %d clientes, ref_date=%s", len(features), ref_date.date())
    return features


def asignar_clusters(features: pd.DataFrame) -> pd.DataFrame:
    """K-Means k=4 y asignación automática de nombres."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features[FEATURE_COLS])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    features = features.copy()
    features["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Asignar nombres por centroides
    centroids = features.groupby("cluster_id")[FEATURE_COLS].mean()
    sizes = features["cluster_id"].value_counts()

    ballena = centroids["ticket_promedio"].idxmax()
    reposicion = centroids.drop(ballena)["frecuencia_total"].idxmax()
    resto = [c for c in centroids.index if c not in [ballena, reposicion]]
    masa_critica = sizes[resto].idxmax()
    intermedio = [c for c in resto if c != masa_critica][0]

    nombres = {
        reposicion: _SEGMENT_DESCRIPTIONS["reposicion"],
        intermedio: _SEGMENT_DESCRIPTIONS["intermedio"],
        masa_critica: _SEGMENT_DESCRIPTIONS["masa_critica"],
        ballena: _SEGMENT_DESCRIPTIONS["ballenas"],
    }
    features["cluster_descripcion"] = features["cluster_id"].map(nombres)

    for cid, desc in nombres.items():
        n = (features["cluster_id"] == cid).sum()
        tp = centroids.loc[cid, "ticket_promedio"]
        logger.info("  Cluster %d: %s — %d clientes, ticket $%.0f", cid, desc.split("—")[0].strip(), n, tp)

    return features


# ---------------------------------------------------------------------------
#  4. Punto de venta + contacto
# ---------------------------------------------------------------------------


def punto_de_venta_principal(df: pd.DataFrame) -> pd.Series:
    """Punto de venta más frecuente por cliente."""
    return (
        df.groupby("user_id")["id_point_sale"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("id_point_sale")
    )


def obtener_contactos(user_ids: list[str]) -> pd.DataFrame:
    """Consulta API de terceros para datos de contacto."""
    from api_client import fetch_third_parties, get_auth_token

    logger.info("Consultando API de terceros (%d clientes)...", len(user_ids))
    token = get_auth_token()
    records = fetch_third_parties(token, user_ids, batch_size=10)

    if not records:
        logger.warning("Sin respuesta de API de terceros")
        return pd.DataFrame(columns=["user_id"])

    df = pd.DataFrame(records)
    rename = {
        "document": "user_id", "name": "name", "email": "email",
        "cellphone": "phone", "document_type": "document_type",
        "country": "country", "department": "department",
        "town": "town", "gender": "gender",
    }
    rename = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=rename)

    cols = [v for v in rename.values() if v in df.columns]
    df = df[cols].drop_duplicates(subset=["user_id"])
    df["user_id"] = df["user_id"].astype(str).str.strip()

    logger.info("Contactos obtenidos: %d", len(df))
    return df


# ---------------------------------------------------------------------------
#  5. Pipeline principal
# ---------------------------------------------------------------------------


def segmentar(sin_contacto: bool = False, output: str | None = None) -> Path:
    """Pipeline completo: carga → limpieza → features → clusters → exporta."""
    output_path = Path(output) if output else PROCESSED_DIR / "clientes_segmentados.parquet"

    # Cargar y limpiar
    df = cargar_datos_vip()
    df = limpiar_ids(df)

    # Features y clustering
    features = calcular_features(df)
    features = asignar_clusters(features)

    # Armar tabla de exportación
    df_export = features[["cluster_id", "cluster_descripcion"]].reset_index()
    df_export = df_export.merge(punto_de_venta_principal(df), on="user_id", how="left")

    # Contacto (opcional)
    if not sin_contacto:
        contactos = obtener_contactos(df_export["user_id"].tolist())
        df_export = df_export.merge(contactos, on="user_id", how="left")
    else:
        for col in ["name", "email", "phone", "document_type", "country", "department", "town", "gender"]:
            df_export[col] = None
        logger.info("Exportando sin datos de contacto (--sin-contacto)")

    # Ordenar columnas
    col_order = [
        "user_id", "cluster_id", "cluster_descripcion", "id_point_sale",
        "name", "email", "phone", "document_type",
        "country", "department", "town", "gender",
    ]
    df_export = df_export[[c for c in col_order if c in df_export.columns]]

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_parquet(output_path, index=False)
    logger.info("Guardado %s: %d clientes", output_path, len(df_export))

    return output_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmenta clientes VIP y exporta parquet.")
    parser.add_argument("--sin-contacto", action="store_true",
                        help="No consultar API de terceros (solo cluster + punto de venta)")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta del parquet de salida (default: data/processed/clientes_segmentados.parquet)")
    args = parser.parse_args()

    path = segmentar(sin_contacto=args.sin_contacto, output=args.output)
    print(f"\nListo: {path}")
