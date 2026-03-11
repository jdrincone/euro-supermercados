"""Módulo de compatibilidad — re-exporta desde los nuevos módulos.

.. deprecated::
    Importar directamente desde ``config``, ``api_client`` o ``data_io``.
"""

from api_client import fetch_sales as obtener_ventas  # noqa: F401
from api_client import fetch_third_parties as obtener_terceros  # noqa: F401
from api_client import get_auth_token as obtener_token  # noqa: F401
from config import load_config as read_yaml  # noqa: F401
