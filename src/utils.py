
import requests
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict


import yaml

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def obtener_token(username, password):
    url = "https://back-middleware.eurosupermercados.com/api-auth/"
    payload = {"username": username, "password": password}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["key"]


def obtener_terceros(token, documentos, batch_size=10):
    url = "https://back-middleware.eurosupermercados.com/euro_nous_thirdparties/thirdparties/"
    headers = {"Authorization": f"key {token}", "Content-Type": "application/json"}
    terceros_totales = []
    terceros_faltantes = []

    # Dividir documentos en batches
    for i in range(0, len(documentos), batch_size):
        batch = documentos[i:i + batch_size]
        params = {"document__in": ",".join(batch)}

        print(f"Consultando terceros: {i+1} al {i+len(batch)} de {len(documentos)}")
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            terceros_totales.extend(data)
        except Exception as e:
            print(f"Error en el batch {batch}")
            terceros_faltantes.extend(batch)


    return terceros_totales


def obtener_ventas(token, fecha_inicio, fecha_fin):
    url = "https://back-middleware.eurosupermercados.com/euro_nous_thirdparties/sales/get_sales/"
    headers = {"Authorization": f"key {token}", "Content-Type": "application/json"}

    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    ventas_totales = []
    dias_faltantes = []

    fecha_actual = fecha_inicio
    while fecha_actual < fecha_fin:

        fecha_str = fecha_actual.strftime('%d/%m/%Y')
        params = {"date__date_sale": fecha_str}

        print(f"Consultando ventas del día: {fecha_str}")
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            ventas_totales.extend(data)

        except Exception as e:
            print(f"Error en el día {fecha_str}")
            dias_faltantes.extend(fecha_str)

        fecha_actual += timedelta(days=1)

    return ventas_totales, dias_faltantes




def read_yaml(path: str | Path) -> Dict[str, Any]:
    """Carga un archivo YAML y lo devuelve como dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)