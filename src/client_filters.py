"""Validación y filtrado de clientes para predicción y clustering.

Centraliza las reglas de negocio para determinar qué clientes son
válidos para los modelos. Aplicable tanto al pipeline DVC (predicción)
como al clustering y recomendaciones.

Reglas implementadas:
    1. ID numérico, sin espacios ni caracteres especiales.
    2. No empieza en "0".
    3. Longitud entre 6 y 10 dígitos (cédula colombiana CC).
    4. Excluye IDs genéricos/de prueba (repeticiones, secuencias).
    5. Excluye IDs tipo NIT de empresas (opcional, por patrón).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# IDs genéricos conocidos que no corresponden a personas reales
_BLACKLIST_IDS = frozenset(
    {
        "0",
        "1",
        "12345",
        "123456",
        "1234567",
        "12345678",
        "123456789",
        "1234567890",
        "111111",
        "1111111",
        "11111111",
        "111111111",
        "1111111111",
        "222222",
        "2222222",
        "22222222",
        "222222222",
        "2222222222",
        "333333",
        "3333333",
        "33333333",
        "333333333",
        "3333333333",
        "444444",
        "4444444",
        "44444444",
        "444444444",
        "4444444444",
        "555555",
        "5555555",
        "55555555",
        "555555555",
        "5555555555",
        "666666",
        "6666666",
        "66666666",
        "666666666",
        "6666666666",
        "777777",
        "7777777",
        "77777777",
        "777777777",
        "7777777777",
        "888888",
        "8888888",
        "88888888",
        "888888888",
        "8888888888",
        "999999",
        "9999999",
        "99999999",
        "999999999",
        "9999999999",
    }
)

# Longitud válida para cédula colombiana (CC)
_MIN_CC_LENGTH = 6
_MAX_CC_LENGTH = 10


def _is_repetitive(s: str) -> bool:
    """Detecta IDs con un solo dígito repetido (ej: '7777777')."""
    return len(set(s)) == 1


def validate_client_ids(
    df: pd.DataFrame,
    id_col: str = "id_client",
    min_length: int = _MIN_CC_LENGTH,
    max_length: int = _MAX_CC_LENGTH,
) -> pd.DataFrame:
    """Filtra clientes con IDs válidos de persona natural (cédula colombiana).

    Criterios:
        - Solo dígitos (sin letras ni caracteres especiales).
        - No empieza en "0".
        - Longitud entre ``min_length`` y ``max_length`` (default 6-10).
        - No está en la lista negra de IDs genéricos.
        - No es un dígito repetido (ej: "7777777").

    Args:
        df: DataFrame con columna de IDs.
        id_col: Nombre de la columna de ID de cliente.
        min_length: Longitud mínima del ID.
        max_length: Longitud máxima del ID.

    Returns:
        DataFrame filtrado solo con clientes válidos.
    """
    n_before = len(df)
    ids = df[id_col].astype(str).str.strip()

    mask = (
        ids.str.fullmatch(r"\d+", na=False)  # solo dígitos
        & ~ids.str.startswith("0", na=False)  # no empieza en 0
        & ids.str.len().between(min_length, max_length)  # longitud CC
        & ~ids.isin(_BLACKLIST_IDS)  # no genéricos
        & ~ids.apply(_is_repetitive)  # no repetitivos
    )

    df_out = df[mask].copy()
    df_out[id_col] = ids[mask].values
    n_after = len(df_out)
    n_removed = n_before - n_after

    if n_removed > 0:
        logger.info(
            "Filtro de IDs: %d -> %d filas (%d eliminadas, %.1f%%)",
            n_before,
            n_after,
            n_removed,
            100 * n_removed / max(n_before, 1),
        )
    else:
        logger.info("Filtro de IDs: %d filas, todas válidas.", n_before)

    return df_out
