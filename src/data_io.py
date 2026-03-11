"""Funciones compartidas de carga y guardado de datos.

Centraliza operaciones de I/O repetidas en múltiples scripts:
lectura/escritura de Parquet, carga de catálogo de productos,
carga del calendario de features y del modelo calibrado.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Parquet genérico
# ---------------------------------------------------------------------------


def load_parquet(path: Path, label: str = "") -> pd.DataFrame:
    """Carga un archivo Parquet y reporta tamaño.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {path.resolve()}")
    df = pd.read_parquet(path)
    tag = label or path.stem
    logger.info("%s cargado: %d filas desde %s", tag, len(df), path)
    return df


def save_parquet(df: pd.DataFrame, path: Path, label: str = "") -> None:
    """Guarda un DataFrame en Parquet, creando directorios si es necesario."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    tag = label or path.stem
    logger.info("%s guardado: %d filas en %s", tag, len(df), path)


# ---------------------------------------------------------------------------
#  Catálogo de productos
# ---------------------------------------------------------------------------


def load_product_catalog(path: Path) -> pd.DataFrame:
    """Carga catálogo de productos con columnas normalizadas.

    Devuelve un DataFrame con columnas ``['product', 'description']``
    sin duplicados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        KeyError: Si faltan columnas requeridas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Catálogo de productos no encontrado: {path.resolve()}"
        )
    df = pd.read_csv(path, dtype={"codigo_unico": str})

    required_cols = {"codigo_unico", "description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Columnas faltantes en catálogo {path}: {missing}")

    df = (
        df.rename(columns={"codigo_unico": "product"})
        .assign(product=lambda d: d["product"].str.strip())
        .loc[:, ["product", "description"]]
        .drop_duplicates("product")
    )
    logger.info("Catálogo cargado: %d productos únicos", len(df))
    return df


# ---------------------------------------------------------------------------
#  Calendario de features + modelo calibrado
# ---------------------------------------------------------------------------


def load_calendar_features(cfg: dict[str, Any]) -> pd.DataFrame:
    """Carga calendario de features con tipos normalizados.

    Asegura que ``date`` es datetime normalizado y ``client`` es string.
    """
    from config import processed_path

    path = processed_path(cfg) / cfg["featurize"]["output_file"]
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["client"] = df["client"].astype(str)
    logger.info("Calendario cargado: %d filas desde %s", len(df), path)
    return df


def load_calibrated_model(cfg: dict[str, Any]) -> Any:
    """Carga el modelo calibrado (``CalibratedClassifierCV``).

    Raises:
        FileNotFoundError: Si el modelo no existe (ejecutar ``dvc repro`` primero).
    """
    from config import model_dir

    path = model_dir(cfg) / cfg["model"]["calibrated_model_name"]
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo calibrado no encontrado: {path.resolve()}. "
            "Ejecuta `dvc repro` para generar el modelo."
        )
    model = joblib.load(path)
    logger.info("Modelo calibrado cargado: %s", path)
    return model


# ---------------------------------------------------------------------------
#  Modelos genérico
# ---------------------------------------------------------------------------


def save_model(model: Any, path: Path) -> None:
    """Serializa un modelo con joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modelo guardado: %s", path)


# ---------------------------------------------------------------------------
#  Reportes
# ---------------------------------------------------------------------------


def save_text_report(path: Path, header: str, content: str) -> None:
    """Guarda un reporte de texto con encabezado subrayado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{header}\n{'=' * len(header)}\n{content}")
    logger.info("Reporte guardado: %s", path)


def save_json(path: Path, obj: dict) -> None:
    """Guarda un diccionario como JSON indentado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4)
    logger.info("JSON guardado: %s", path)
