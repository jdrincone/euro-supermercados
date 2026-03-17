# Segmentación de Clientes VIP — Euro Supermercados

## Resumen

Segmentamos los clientes del programa VIP en **4 grupos** usando K-Means sobre su comportamiento de compra. Cada cliente queda asignado a un cluster con su punto de venta principal y datos de contacto.

> **Nota:** Este sistema (k=4, datos VIP) es independiente del pipeline DVC de recomendaciones
> (k=5, datos de `initial_sales_clean.parquet`). Son análisis complementarios con datos y
> objetivos distintos.

## Los 4 Segmentos

| Cluster | Nombre | Clientes | Ticket Promedio | Descripción |
|---------|--------|----------|----------------|-------------|
| Reposición | ~2,100 | ~$51,000 | Compras frecuentes, ticket bajo. ~220 visitas/año. |
| Mercado Intermedio | ~7,800 | ~$84,000 | Frecuencia media (~21 visitas/año), riesgo de pérdida. |
| Masa Crítica | ~53,600 | ~$93,000 | Mayor volumen de clientes (73%), foco de crecimiento. |
| Ballenas | ~10,000 | ~$390,000 | Ticket alto. Minisupermercados o negocios. |

### Criterios de asignación automática

Los nombres se asignan automáticamente según los centroides de K-Means:

1. **Ballenas** = cluster con mayor `ticket_promedio`
2. **Reposición** = cluster con mayor `frecuencia_total` (excluyendo Ballenas)
3. **Masa Crítica** = cluster con más clientes (de los restantes)
4. **Mercado Intermedio** = el restante

## Archivo de Salida

```
data/processed/clientes_segmentados.parquet
```

| Columna | Tipo | Ejemplo |
|---------|------|---------|
| `user_id` | String | `1128446657` |
| `cluster_id` | Int64 | `3` |
| `cluster_nombre` | String | `Masa Crítica` |
| `cluster_descripcion` | String | `Mayor volumen de clientes (73%), foco de crecimiento.` |
| `id_point_sale` | String | `FRO` |
| `name` | String | `Juan Pérez` |
| `email` | String | `juan@email.com` |
| `phone` | String | `3001234567` |
| `document_type` | String | `CC` |
| `country` | String | `Colombia` |
| `department` | String | `Valle del Cauca` |
| `town` | String | `Cali` |
| `gender` | String | `M` |

## Cómo Ejecutar

### Script (producción)

```bash
# Solo segmentación (rápido, ~25 segundos)
uv run python src/segmentar_clientes.py --sin-contacto

# Con datos de contacto desde la API (varios minutos)
uv run python src/segmentar_clientes.py
```

### Notebook (exploración)

Abrir `notebooks/02_segmentacion_clientes.ipynb` y ejecutar todas las celdas.

Incluye visualizaciones adicionales:
- Evolución mensual por cluster
- Canasta "de combate" (productos top por penetración)
- Ciclo de gasto por día del mes (efecto nómina/quincena)
- Detección de clientes en caída (P1 Ene-Jul vs P2 Ago-Dic)
- "Efecto recorte" (qué productos dejaron de comprar)

## Datos de Entrada

El análisis usa `data/processed/df_vip.parquet` (o `df_vip.csv`).

Este archivo contiene transacciones del programa de fidelización VIP: solo clientes
con actividad en al menos 10 semanas distintas.

Si no existe, descárgalo con:

```bash
uv run python src/download_vip.py --download --desde 2025-01-01 --hasta 2026-03-17 --process
```

## Metodología

1. **Limpieza de IDs** — `src/client_filters.py`: cédula colombiana válida (6-10 dígitos, sin genéricos)
2. **6 features por cliente** (sobre tickets únicos, `tiket_price` = valor total del ticket):
   - `ticket_promedio`: mean del ticket
   - `valor_tipico`: mediana del ticket
   - `gasto_total_vida`: suma total gastada
   - `volatilidad_gasto`: desviación estándar del ticket
   - `frecuencia_total`: número de visitas
   - `dias_sin_comprar`: días desde la última compra
3. **K-Means k=4** — StandardScaler + validación por método del codo y silhouette score
4. **Asignación automática** de nombres según centroides

## Estructura de Archivos

```
src/
  segmentar_clientes.py       # Script de producción (k=4, datos VIP)
  download_vip.py             # Descarga datos VIP desde la API
  api_client.py               # Cliente HTTP (auth + retries)
  client_filters.py           # Validación centralizada de cédulas

notebooks/
  02_segmentacion_clientes.ipynb   # Análisis exploratorio completo

data/processed/
  df_vip.parquet                   # Entrada: transacciones VIP
  clientes_segmentados.parquet     # Salida: clientes con cluster asignado
```
