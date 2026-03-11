# Euro ML

Predicción de probabilidad de compra diaria y recomendación de productos para Euro Supermercados.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (instala Python 3.13 automáticamente)

## Setup

```bash
uv sync
cp .env.example .env  # Configurar API_USERNAME y API_PASSWORD
```

Datos semilla requeridos en `data/processed/`:
- `initial_sales_clean.parquet` — Histórico de ventas
- `productos.csv` — Catálogo de productos

## Pipeline DVC

```bash
uv run dvc repro          # Primera vez
uv run dvc repro --force  # Re-entrenamiento
```

Etapas: `load` → `preprocess` → `featurize` → `train` → `evaluate` → `backtest`

Ejecución individual (si ya tienes los datos semilla, saltar `load`):

```bash
uv run python src/preprocess.py --config params.yaml
uv run python src/featurize.py --config params.yaml
uv run python src/train.py --config params.yaml
uv run python src/evaluate.py --config params.yaml
```

## Recomendadores

Entrenamiento manual (fuera del pipeline DVC):

```bash
uv run python src/train_recommender.py --config params.yaml             # Item-Item (top 10% global)
uv run python src/train_recommender_by_client.py --config params.yaml   # Item-Item (top 5/cliente)
uv run python src/train_recommender_by_clustering.py --config params.yaml  # K-Means por segmentos
```

## Predicción

```bash
uv run python src/predict.py --dates 2025-11-07 2025-11-08
uv run python src/predict.py --dates 2025-11-07 --threshold 0.6 --output pred.csv
```

Unir predicciones con recomendaciones:

```bash
uv run python src/get_recommendations.py \
    --input_file predictions/predictions_client.csv \
    --output_file predictions/preds_with_recs.csv \
    --include_clustering
```

## Backtesting

Ajustar `backtesting.backtest_start_date` / `backtest_end_date` en `params.yaml`:

```bash
uv run python src/backtest.py --config params.yaml
```

## Estructura

```
src/
├── config.py            # Carga YAML + resolución de rutas
├── api_client.py        # Cliente HTTP para API Euro
├── data_io.py           # I/O: Parquet, modelos, reportes
├── patterns.py          # Patrones de compra (preprocess + clustering)
├── collaborative.py     # Filtrado colaborativo item-item
├── load_data.py         # Descarga ventas desde API
├── preprocess.py        # Filtrado de clientes por patrones
├── featurize.py         # Features temporales + rolling windows
├── train.py             # Entrena LogisticRegression
├── evaluate.py          # Calibración sigmoid + métricas + SHAP
├── backtest.py          # Backtesting con ventas reales
├── predict.py           # Inferencia + recomendaciones por historial
├── train_recommender.py             # Item-Item CF (top 10%)
├── train_recommender_by_client.py   # Item-Item CF (top 5/cliente)
├── train_recommender_by_clustering.py  # K-Means clustering
└── get_recommendations.py           # Une predicciones + recomendaciones
```

## Configuración

Todo en `params.yaml`:

| Sección | Descripción |
|---------|-------------|
| `data` | Rutas y ventana temporal |
| `preprocess` | Filtros de patrones de compra |
| `featurize` | Features temporales y rolling windows |
| `train` | Hiperparámetros LogisticRegression |
| `evaluate` | Calibración y métricas |
| `recommendations_item_item` | Recomendador colaborativo |
| `recommendations_clustering` | Recomendador por segmentos |

## Artefactos

| Archivo | Descripción |
|---------|-------------|
| `models/model.joblib` | Modelo base |
| `models/calibrated_model.joblib` | Modelo calibrado (sigmoid) |
| `reports/metrics.json` | ROC-AUC, Brier, Precision, Recall |
| `reports/plots/` | Calibración, importancia, SHAP |
| `data/processed/recommendations/` | Recomendaciones precalculadas |
