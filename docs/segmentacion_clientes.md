# Segmentación de Clientes VIP — Euro Supermercados

## Resumen

Segmentamos los clientes del programa VIP en **4 grupos** usando K-Means sobre su comportamiento de compra. Cada cliente queda asignado a un cluster con su punto de venta principal y datos de contacto.

## Los 4 Segmentos

| Cluster | Nombre | Clientes | Ticket Promedio | Qué son |
|---------|--------|----------|----------------|---------|
| **Reposición** | ~2,100 | ~$51,000 | Compran muy seguido pero poco. Visitas de reposición diaria (pan, verduras). ~220 visitas/año. |
| **Mercado Intermedio** | ~7,800 | ~$84,000 | Compran poco (~21 visitas/año). Muchos dejaron de venir. Riesgo de pérdida. |
| **Masa Crítica** | ~53,600 | ~$93,000 | El grupo más grande (73% de clientes). Familias que mercan regularmente. Foco de crecimiento. |
| **Ballenas** | ~10,000 | ~$390,000 | Ticket muy alto. Probablemente minisupermercados o negocios que revenden. |

## Archivo de Salida

```
data/processed/clientes_segmentados.parquet
```

| Columna | Ejemplo | Descripción |
|---------|---------|-------------|
| `user_id` | `1128446657` | Cédula del cliente |
| `cluster_id` | `3` | Número del cluster (0-3) |
| `cluster_descripcion` | `Masa Crítica — ...` | Nombre y descripción del segmento |
| `id_point_sale` | `FRO` | Tienda donde más compra |
| `name` | `Juan Pérez` | Nombre (de la API) |
| `email` | `juan@...` | Email |
| `phone` | `300...` | Teléfono |
| `document_type` | `CC` | Tipo de documento |
| `country` | `Colombia` | País |
| `department` | `Valle del Cauca` | Departamento |
| `town` | `Cali` | Municipio |
| `gender` | `M` | Género |

## Cómo Ejecutar

### Opción 1: Script (recomendado para producción)

```bash
# Solo segmentación (rápido, ~25 segundos)
uv run python src/segmentar_clientes.py --sin-contacto

# Con datos de contacto desde la API (toma varios minutos)
uv run python src/segmentar_clientes.py
```

### Opción 2: Notebook (para exploración y análisis)

Abrir `notebooks/02_segmentacion_clientes.ipynb` y ejecutar todas las celdas.

El notebook incluye visualizaciones adicionales:
- Evolución mensual por cluster
- Canasta "de combate" (productos más vendidos por grupo)
- Ciclo de gasto por día del mes (efecto nómina/quincena)
- Detección de clientes en caída (P1 vs P2)
- "Efecto recorte" (qué productos dejaron de comprar)

## Datos de Entrada

El análisis usa `data/processed/df_vip.parquet` (o `df_vip.csv`).

Si no existe, descárgalo con:

```bash
# Descargar ventas de Ene-2025 a hoy + procesar
uv run python src/download_vip.py --download --desde 2025-01-01 --hasta 2026-03-17 --process
```

## Metodología

1. **Limpieza de IDs** — Solo cédulas colombianas válidas (6-10 dígitos, sin genéricos como `111111`)
2. **6 features por cliente** calculadas sobre tickets únicos:
   - Ticket promedio y mediana
   - Gasto total de vida
   - Volatilidad del gasto (desviación estándar)
   - Frecuencia total (número de visitas)
   - Días sin comprar (desde la última visita)
3. **K-Means con k=4** — StandardScaler + selección validada por método del codo y silhouette
4. **Asignación automática de nombres** según centroides (el de mayor ticket = Ballena, el de mayor frecuencia = Reposición, etc.)

## Estructura de Archivos

```
src/
  segmentar_clientes.py    # Script de producción
  download_vip.py          # Descarga datos VIP desde la API
  api_client.py            # Cliente HTTP (auth + retries)

notebooks/
  02_segmentacion_clientes.ipynb  # Análisis exploratorio completo

data/processed/
  df_vip.parquet                  # Datos de entrada (transacciones VIP)
  clientes_segmentados.parquet    # Salida: clientes con cluster asignado
```
