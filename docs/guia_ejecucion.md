# Guia de Ejecucion

Como instalar, ejecutar y operar el sistema.

---

## 1. Requisitos

- **Python 3.13+**
- **uv** (gestor de paquetes): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **DVC** (incluido en dependencias)
- **Credenciales API** en archivo `.env` (para `load_data`, `predict`, `backtest`):
  ```
  API_USERNAME=tu_usuario
  API_PASSWORD=tu_password
  ```

---

## 2. Instalacion

```bash
# Clonar repositorio
git clone <repo-url> && cd euro

# Instalar dependencias
uv sync
```

Esto instala todas las dependencias definidas en `pyproject.toml`.

---

## 3. Pipeline DVC Completo

Ejecuta las 6 etapas en orden: `load` → `preprocess` → `featurize` → `train` → `evaluate` → `backtest`.

```bash
# Ejecutar (solo etapas con cambios)
uv run dvc repro

# Forzar re-ejecucion completa
uv run dvc repro --force
```

DVC detecta automaticamente que etapas necesitan re-ejecutarse segun cambios en deps, params o codigo.

---

## 4. Etapas Individuales

Cada script acepta `--config params.yaml`.

### 4.1 Carga de datos (requiere API)
```bash
uv run python src/load_data.py --config params.yaml
```
Descarga ventas nuevas de la API, limpia IDs, deduplica y guarda `initial_sales_clean.parquet`.

### 4.2 Preprocesamiento
```bash
uv run python src/preprocess.py --config params.yaml
```
Filtra clientes validos y activos, genera `daily.parquet` y `filtered_agg_sales_for_rec.parquet`.

### 4.3 Features
```bash
uv run python src/featurize.py --config params.yaml
```
Genera calendario completo con 13 features. Salida: `calendar_features.parquet`.

### 4.4 Entrenamiento
```bash
uv run python src/train.py --config params.yaml
```
Entrena modelo (configurable en `train.model_type`). Salida: `models/model.joblib`.

### 4.5 Evaluacion
```bash
uv run python src/evaluate.py --config params.yaml
```
Calibra modelo, genera metricas, reportes y graficos. Salida principal: `models/calibrated_model.joblib`.

### 4.6 Backtesting (requiere API)
```bash
uv run python src/backtest.py --config params.yaml
```
Compara predicciones con ventas reales. Salida: `reports/backtesting_metrics.csv`.

---

## 5. Predicciones Operativas

Genera predicciones para fechas especificas con recomendaciones y datos de contacto.

```bash
# Una fecha
uv run python src/predict.py --dates 2025-11-07

# Varias fechas
uv run python src/predict.py --dates 2025-11-07 2025-11-08 2025-11-09

# Con umbral personalizado
uv run python src/predict.py --dates 2025-11-07 --threshold 0.6

# Con nombre de archivo de salida
uv run python src/predict.py --dates 2025-11-07 --output pred_nov7.csv
```

Salida: CSV en `predictions/` con columnas: `client`, `date`, `probability`, `phone`, `email`, `recommendations`.

---

## 6. Recomendadores (Manuales)

No estan en el pipeline DVC. Se ejecutan independientemente.

### 6.1 Item-item global (top 10% productos)
```bash
uv run python src/train_recommender.py --config params.yaml
```
Genera: `data/processed/recommendations/precomputed_item_item_recs.parquet`

### 6.2 Item-item por cliente (top 5 productos/cliente)
```bash
uv run python src/train_recommender_by_client.py --config params.yaml
```

### 6.3 Segmentacion + recomendaciones por cluster
```bash
uv run python src/train_recommender_by_clustering.py --config params.yaml
```
Genera:
- `data/processed/cluster_assignments.parquet` (asignacion de segmento por cliente)
- `data/processed/recommendations/precomputed_cluster_recs.parquet` (recomendaciones por segmento)

### 6.4 Unir recomendaciones con predicciones
```bash
# Solo item-item
uv run python src/get_recommendations.py \
    --input_file predictions/predictions_client.csv \
    --output_file predictions/preds_with_recs.csv

# Con clustering
uv run python src/get_recommendations.py \
    --input_file predictions/predictions_client.csv \
    --output_file predictions/preds_with_recs.csv \
    --include_clustering
```

---

## 7. Notebooks

Ubicados en `notebooks/`. Ejecutar con JupyterLab:

```bash
uv run jupyter lab
```

| Notebook | Descripcion |
|----------|-------------|
| `01_pipeline_predictivo.ipynb` | Pipeline predictivo interactivo |
| `02_segmentacion_clientes.ipynb` | Segmentacion en 5 perfiles con graficos |
| `03_recomendador_colaborativo.ipynb` | Filtrado colaborativo item-item |

Los notebooks importan modulos de `src/` y usan los mismos datos centrales.

---

## 8. Flujos de Trabajo Comunes

### Actualizar modelo con datos nuevos
```bash
# 1. Descargar ventas recientes
uv run python src/load_data.py --config params.yaml

# 2. Re-entrenar pipeline completo
uv run dvc repro

# 3. Generar predicciones para manana
uv run python src/predict.py --dates 2025-11-10
```

### Re-segmentar clientes
```bash
# 1. Asegurarse de tener datos actualizados
uv run python src/preprocess.py --config params.yaml

# 2. Ejecutar segmentacion
uv run python src/train_recommender_by_clustering.py --config params.yaml

# 3. Unir con predicciones
uv run python src/get_recommendations.py \
    --input_file predictions/predictions_client.csv \
    --output_file predictions/preds_with_recs.csv \
    --include_clustering
```

### Cambiar tipo de modelo
Editar `params.yaml`:
```yaml
train:
  model_type: logistic_regression  # o hist_gradient_boosting
```
Luego re-entrenar:
```bash
uv run dvc repro -s train
```

### Ver metricas del modelo
```bash
# Metricas JSON
cat reports/metrics.json

# Comparar con version anterior
uv run dvc metrics diff
```

---

## 9. Verificacion de Calidad

```bash
# Linter
uv run ruff check src/

# Formato
uv run ruff format src/

# Ver estado del pipeline
uv run dvc status
```

---

## 10. Archivos de Datos Requeridos

El sistema necesita dos archivos semilla en `data/processed/`:

1. **`initial_sales_clean.parquet`** — Se genera con `load_data.py` (requiere API) o se coloca manualmente.
2. **`productos.csv`** — Catalogo de productos. Se coloca manualmente.

Todo lo demas se genera automaticamente a partir de estos dos archivos.
