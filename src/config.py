"""Gestión centralizada de configuración y resolución de rutas.

Todas las rutas del proyecto se derivan de ``params.yaml`` para evitar
construcciones manuales dispersas en cada script.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Carga un archivo YAML de configuración y lo devuelve como dict.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el YAML es inválido.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {path.resolve()}"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error al parsear {path}: {exc}") from exc

    # Validar secciones mínimas requeridas
    required_sections = ["data", "base"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise KeyError(f"Secciones faltantes en {path}: {missing}")

    logger.debug("Config cargada: %d secciones desde %s", len(cfg), path)
    return cfg


def processed_path(cfg: dict[str, Any]) -> Path:
    """Directorio de datos procesados: ``data.base_path / data.processed_folder``."""
    return Path(cfg["data"]["base_path"]) / cfg["data"]["processed_folder"]


def model_dir(cfg: dict[str, Any]) -> Path:
    """Directorio de modelos: ``model.model_dir``."""
    return Path(cfg["model"]["model_dir"])


def reports_dir(cfg: dict[str, Any]) -> Path:
    """Directorio de reportes: ``reports.reports_dir``."""
    return Path(cfg["reports"]["reports_dir"])


def plots_dir(cfg: dict[str, Any]) -> Path:
    """Directorio de gráficos dentro de reportes."""
    return reports_dir(cfg) / cfg["reports"]["plots_dir"]
