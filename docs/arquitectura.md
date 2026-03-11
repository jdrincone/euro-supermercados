# Arquitectura del Sistema

## Vision General

El sistema tiene tres componentes principales:

1. **Modelo predictivo** -- Estima la probabilidad diaria de compra por cliente.
2. **Motor de recomendaciones** -- Tres estrategias: historial reciente, filtrado colaborativo item-item y clustering.
3. **Segmentacion de clientes** -- Clasifica clientes en 5 perfiles de negocio.

Todo el codigo esta en `src/`, la configuracion en `params.yaml`, y los notebooks interactivos en `notebooks/`.

---

## Estructura del Repositorio

```
euro/
├── src/                          # Codigo fuente (modulos Python)
│   ├── config.py                 # Carga de YAML y resolucion de rutas
│   ├── api_client.py             # Cliente HTTP para API Euro middleware
│   ├── data_io.py                # I/O de Parquet, modelos, reportes
│   ├── client_filters.py         # Validacion de cedulas colombianas
│   ├── patterns.py               # Patrones de compra y features de segmentacion
│   ├── collaborative.py          # Filtrado colaborativo item-item
│   ├── load_data.py              # Etapa DVC: sincroniza ventas con API
│   ├── preprocess.py             # Etapa DVC: filtra y agrega datos
│   ├── featurize.py              # Etapa DVC: ingenieria de features
│   ├── train.py                  # Etapa DVC: entrena modelo
│   ├── evaluate.py               # Etapa DVC: calibra y evalua
│   ├── backtest.py               # Etapa DVC: backtesting con datos reales
│   ├── predict.py                # Genera predicciones + recomendaciones
│   ├── train_recommender.py      # Recomendador item-item (top 10% global)
│   ├── train_recommender_by_client.py   # Recomendador item-item (top 5/cliente)
│   ├── train_recommender_by_clustering.py  # Segmentacion + recs por cluster
│   └── get_recommendations.py    # Une recomendaciones con predicciones
│
├── notebooks/                    # Notebooks interactivos (Jupyter)
│   ├── 01_pipeline_predictivo.ipynb
│   ├── 02_segmentacion_clientes.ipynb
│   └── 03_recomendador_colaborativo.ipynb
│
├── data/
│   ├── processed/                # Datos procesados (pipeline)
│   │   ├── initial_sales_clean.parquet  # Ventas historicas (dato origen)
│   │   ├── productos.csv                # Catalogo de productos (dato origen)
│   │   ├── daily.parquet                # Cliente-dia (featurize input)
│   │   ├── filtered_agg_sales_for_rec.parquet  # Para recomendadores
│   │   ├── calendar_features.parquet    # Features completas (train input)
│   │   ├── cluster_assignments.parquet  # Asignaciones de segmento
│   │   └── recommendations/             # Recomendaciones precalculadas
│   └── raw/
│       └── productos.csv               # Catalogo original
│
├── models/                       # Modelos serializados (joblib)
├── predictions/                  # CSVs de predicciones
├── reports/                      # Metricas, reportes y graficos
├── params.yaml                   # Configuracion central
├── dvc.yaml                      # Definicion del pipeline DVC
└── docs/                         # Esta documentacion
```

---

## Flujo de Datos

```
API Euro Middleware
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  load_data  │────▶│  preprocess  │────▶│  featurize │
│  (API sync) │     │  (filtros)   │     │  (features)│
└─────────────┘     └──────┬───────┘     └─────┬──────┘
       │                   │                   │
       │                   ▼                   ▼
       │            filtered_agg_        calendar_features
       │            sales_for_rec        .parquet
       │            .parquet                   │
       │                   │              ┌────┴─────┐
       ▼                   │              ▼          ▼
initial_sales_clean        │         ┌────────┐ ┌──────────┐
.parquet                   │         │ train  │ │ evaluate │
       │                   │         └───┬────┘ └────┬─────┘
       │                   │             │           │
       │                   ▼             ▼           ▼
       │            ┌──────────────┐  model    calibrated_model
       │            │ Recomendador │  .joblib  .joblib
       │            │  item-item   │              │
       │            └──────────────┘         ┌────┴─────┐
       │                                     ▼          ▼
       ├─────────────────────────────▶  predict    backtest
       │                                  │           │
       ▼                                  ▼           ▼
┌────────────────────┐             predictions/  reports/
│ Segmentacion       │             *.csv         backtesting
│ (clustering)       │                           _metrics.csv
│ 5 perfiles negocio │
└────────┬───────────┘
         ▼
  cluster_assignments.parquet
  precomputed_cluster_recs.parquet
```

---

## Pipeline DVC

Las 6 etapas del pipeline se ejecutan secuencialmente con `uv run dvc repro`:

| Etapa | Script | Input | Output |
|-------|--------|-------|--------|
| `load` | `load_data.py` | API | `initial_sales_clean.parquet` |
| `preprocess` | `preprocess.py` | initial_sales_clean | `daily.parquet`, `filtered_agg_sales_for_rec.parquet` |
| `featurize` | `featurize.py` | daily.parquet | `calendar_features.parquet` |
| `train` | `train.py` | calendar_features | `model.joblib` |
| `evaluate` | `evaluate.py` | model + calendar_features | `calibrated_model.joblib`, metricas, graficos |
| `backtest` | `backtest.py` | calibrated_model + calendar_features + API | `backtesting_metrics.csv` |

Los recomendadores y la segmentacion NO estan en el pipeline DVC (se ejecutan manualmente).

---

## Modulos Compartidos

Seis modulos compartidos evitan duplicacion de logica:

| Modulo | Responsabilidad | Usado por |
|--------|----------------|-----------|
| `config.py` | Carga YAML, resolucion de rutas | Todos |
| `api_client.py` | HTTP con reintentos, auth, descarga ventas | load_data, predict, backtest |
| `data_io.py` | I/O Parquet, modelo, reportes | Todos |
| `client_filters.py` | Validacion de cedulas colombianas | load_data, preprocess, clustering |
| `patterns.py` | Patrones de compra, features segmentacion | preprocess, clustering |
| `collaborative.py` | Matriz sparse, similitud coseno | train_recommender, train_recommender_by_client |

---

## Datos de Origen

Solo dos archivos son la fuente de verdad:

1. **`data/processed/initial_sales_clean.parquet`** -- Ventas historicas descargadas de la API. Columnas: `date_sale`, `id_client`, `product`, `invoice_value_with_discount_and_without_iva`, `amount`.
2. **`data/processed/productos.csv`** -- Catalogo de productos. Columnas: `codigo_unico`, `description`.

Todo lo demas se genera a partir de estos dos archivos.
