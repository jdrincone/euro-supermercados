#!/usr/bin/env python3
"""Descarga y procesa datos VIP de Euro Supermercados.

Dos etapas independientes:
    1. **Descarga**: ventas crudas de la API → un parquet por mes en data/raw/
    2. **Procesamiento**: explota detalles, une catálogo, filtra VIP → data/processed/df_vip.parquet

Uso:
    # Descargar últimos 6 meses (salta meses que ya existen)
    uv run python src/download_vip.py --download --months 6

    # Forzar re-descarga de un mes específico
    uv run python src/download_vip.py --download --desde 2025-06-01 --hasta 2025-06-30

    # Solo procesar (sin descargar)
    uv run python src/download_vip.py --process

    # Descarga + procesamiento completo
    uv run python src/download_vip.py --download --process
"""

import argparse
import ast
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import fetch_sales, get_auth_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Rutas
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "productos.csv"

MIN_WEEKS_VIP = 10  # Mínimo de semanas distintas para ser "VIP" (igual que dowload_vip.py original)


# ---------------------------------------------------------------------------
#  Etapa 1: Descarga
# ---------------------------------------------------------------------------


def download_month(token: str, period: pd.Period, force: bool = False) -> Path:
    """Descarga un mes de ventas y lo guarda como parquet crudo.

    Si el archivo ya existe y force=False, se salta la descarga.
    Retorna la ruta del archivo generado.
    """
    output = RAW_DIR / f"ventas_{period.strftime('%Y-%m')}.parquet"

    if output.exists() and not force:
        logger.info("Ya existe %s — saltando (usar --force para re-descargar)", output.name)
        return output

    inicio = period.start_time.date()
    fin = min(period.end_time.date(), datetime.today().date())

    if inicio > datetime.today().date():
        logger.info("Mes %s es futuro — saltando", period)
        return output

    logger.info("Descargando %s (%s → %s)...", period, inicio, fin)
    records, failed = fetch_sales(token, str(inicio), str(fin + pd.Timedelta(days=1)))

    if failed:
        logger.warning("Días con error en %s: %s", period, failed)

    if not records:
        logger.warning("Sin datos para %s", period)
        return output

    df = pd.DataFrame(records).drop_duplicates(subset=["ID"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Guardado %s: %d tickets", output.name, len(df))

    return output


def download_months(n_months: int = 6, force: bool = False) -> list[Path]:
    """Descarga los últimos n_months meses de ventas."""
    token = get_auth_token()
    hoy = pd.Period(datetime.today(), freq="M")
    periodos = pd.period_range(end=hoy, periods=n_months, freq="M")

    paths = []
    for periodo in periodos:
        path = download_month(token, periodo, force=force)
        paths.append(path)

    return paths


def download_range(desde: str, hasta: str, force: bool = False) -> list[Path]:
    """Descarga ventas en un rango de fechas específico."""
    token = get_auth_token()
    inicio = pd.Period(desde, freq="M")
    fin = pd.Period(hasta, freq="M")
    periodos = pd.period_range(inicio, fin, freq="M")

    paths = []
    for periodo in periodos:
        path = download_month(token, periodo, force=force)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
#  Etapa 2: Procesamiento
# ---------------------------------------------------------------------------


def _parse_details(x):
    """Parsea invoice_details de string a lista de dicts."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []


def _load_catalog() -> pd.DataFrame:
    """Carga el catálogo de productos."""
    if not CATALOG_PATH.exists():
        logger.warning("Catálogo no encontrado en %s — se omite merge de categorías", CATALOG_PATH)
        return pd.DataFrame()

    df = pd.read_csv(CATALOG_PATH, dtype={"codigo_unico": str})
    df = df.rename(columns={"codigo_unico": "product"})
    df["product"] = df["product"].str.strip()
    logger.info("Catálogo cargado: %d productos", len(df))
    return df


def process_raw_to_vip() -> Path:
    """Lee todos los parquets crudos, explota detalles y genera df_vip.parquet.

    Pasos:
        1. Lee todos los archivos data/raw/ventas_*.parquet
        2. Explota invoice_details (JSON anidado) en filas individuales
        3. Une con catálogo de productos (categoría, nombre)
        4. Renombra columnas al esquema VIP
        5. Filtra a clientes recurrentes (>= MIN_WEEKS_VIP semanas distintas)
        6. Guarda como data/processed/df_vip.parquet
    """
    raw_files = sorted(RAW_DIR.glob("ventas_*.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"No se encontraron archivos en {RAW_DIR}/ventas_*.parquet. "
            "Ejecuta primero con --download."
        )

    logger.info("Procesando %d archivos crudos...", len(raw_files))
    dfs = [pd.read_parquet(f) for f in raw_files]
    df = pd.concat(dfs, ignore_index=True)
    df["date_sale"] = pd.to_datetime(df["date_sale"])
    logger.info("Total tickets crudos: %d", len(df))

    # Explotar invoice_details
    df["invoice_details"] = df["invoice_details"].apply(_parse_details)
    df_exp = df.explode("invoice_details").dropna(subset=["invoice_details"]).reset_index(drop=True)
    details = pd.json_normalize(df_exp["invoice_details"])
    df_master = pd.concat(
        [df_exp.drop(columns=["invoice_details"]).reset_index(drop=True), details],
        axis=1,
    )

    # Unir catálogo
    catalog = _load_catalog()
    if not catalog.empty and "product" in df_master.columns:
        df_master["product"] = df_master["product"].astype(str).str.strip()
        df_master = df_master.merge(catalog, on="product", how="left")

    # Renombrar al esquema VIP
    rename_map = {
        "ID": "tiket_id",
        "date_sale": "date_sale",
        "identification_doct": "user_id",
        "invoice_value_without_iva_and_discount": "tiket_price",
        "amount": "amount",
        "category": "category",
        "name": "product_name",
        "tax": "tax",
        "id_point_sale": "id_point_sale",
        "domicilio_status": "domicilio_status",
    }
    # Solo renombrar columnas que existen
    rename_map = {k: v for k, v in rename_map.items() if k in df_master.columns}
    df_master = df_master.rename(columns=rename_map)

    # Seleccionar columnas disponibles
    vip_cols = [
        "tiket_id", "date_sale", "user_id", "tiket_price", "amount",
        "category", "product_name", "tax", "id_point_sale", "domicilio_status",
    ]
    vip_cols = [c for c in vip_cols if c in df_master.columns]
    df_master = df_master[vip_cols].copy()

    # Limpiar tipos
    df_master["date_sale"] = pd.to_datetime(df_master["date_sale"])
    df_master["user_id"] = df_master["user_id"].astype(str).str.strip()

    # Agregar período
    df_master["anio"] = df_master["date_sale"].dt.year
    df_master["semana"] = df_master["date_sale"].dt.isocalendar().week.astype(int)
    df_master["periodo_unico"] = (
        df_master["anio"].astype(str) + "-" + df_master["semana"].astype(str)
    )

    # Filtrar VIP: clientes con actividad en >= MIN_WEEKS_VIP semanas distintas
    semanas_por_usuario = df_master.groupby("user_id")["periodo_unico"].nunique()
    usuarios_vip = semanas_por_usuario[semanas_por_usuario >= MIN_WEEKS_VIP].index

    n_antes = df_master["user_id"].nunique()
    df_vip = df_master[df_master["user_id"].isin(usuarios_vip)].copy()
    n_despues = df_vip["user_id"].nunique()

    logger.info(
        "Filtro VIP (>= %d semanas): %d → %d clientes (%d eliminados)",
        MIN_WEEKS_VIP, n_antes, n_despues, n_antes - n_despues,
    )

    # Guardar
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "df_vip.parquet"
    df_vip.to_parquet(output, index=False)
    logger.info(
        "Guardado %s: %d filas, %d clientes, rango %s → %s",
        output.name,
        len(df_vip),
        n_despues,
        df_vip["date_sale"].min().date(),
        df_vip["date_sale"].max().date(),
    )

    return output


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Descarga y procesa datos VIP de Euro Supermercados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Descargar últimos 6 meses + procesar
  uv run python src/download_vip.py --download --process

  # Solo descargar un rango
  uv run python src/download_vip.py --download --desde 2025-01-01 --hasta 2025-06-30

  # Solo procesar (ya tiene archivos crudos)
  uv run python src/download_vip.py --process

  # Forzar re-descarga
  uv run python src/download_vip.py --download --months 3 --force
        """,
    )
    parser.add_argument("--download", action="store_true", help="Descargar ventas de la API")
    parser.add_argument("--process", action="store_true", help="Procesar crudos → df_vip.parquet")
    parser.add_argument("--months", type=int, default=6, help="Meses a descargar (default: 6)")
    parser.add_argument("--desde", type=str, help="Fecha inicio (YYYY-MM-DD), anula --months")
    parser.add_argument("--hasta", type=str, help="Fecha fin (YYYY-MM-DD), anula --months")
    parser.add_argument("--force", action="store_true", help="Re-descargar meses existentes")

    args = parser.parse_args()

    if not args.download and not args.process:
        parser.print_help()
        sys.exit(1)

    if args.download:
        if args.desde and args.hasta:
            download_range(args.desde, args.hasta, force=args.force)
        else:
            download_months(n_months=args.months, force=args.force)

    if args.process:
        process_raw_to_vip()


if __name__ == "__main__":
    main()
