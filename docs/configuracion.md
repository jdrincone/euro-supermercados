# Guia de Configuracion

Referencia completa de `params.yaml` y `dvc.yaml`.

---

## params.yaml

Archivo central de configuracion. Todos los scripts lo cargan via `config.load_config()`.

### `base`

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `random_state` | int | 42 | Semilla para reproducibilidad (K-Means, splits) |

### `data`

Rutas y estructura de archivos.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `base_path` | str | `data` | Directorio raiz de datos |
| `processed_folder` | str | `processed` | Subcarpeta de datos procesados |
| `predictions_output_folder` | str | `predictions` | Carpeta para CSVs de predicciones |
| `sales_file` | str | `initial_sales_clean.parquet` | Historico de ventas (dato origen) |
| `products_file` | str | `productos.csv` | Catalogo de productos (dato origen) |
| `output_file` | str | `initial_sales_clean.parquet` | Salida de la etapa `load` |
| `output_columns` | list | ver abajo | Columnas del parquet de ventas |
| `months_to_fetch` | int | 6 | Meses de datos a mantener |
| `last_month_with_sale` | int | 3 | Meses recientes para filtrar clientes activos |

Columnas de salida: `date_sale`, `id_client`, `product`, `invoice_value_with_discount_and_without_iva`, `amount`.

### `load_data`

Configuracion de la etapa DVC `load`.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `sales_dtype` | dict | `{id_client: str, product: str}` | Tipos forzados al cargar |
| `output_columns` | list | (mismas que `data.output_columns`) | Columnas de salida |
| `output_file` | str | `initial_sales_clean.parquet` | Archivo de salida |

### `preprocess`

Filtros de calidad de clientes para el modelo predictivo.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `min_purchase_count` | int | 10 | Minimo de dias de compra |
| `max_median_days_between` | int | 35 | Mediana maxima de dias entre compras |
| `max_std_days_between` | int | 3 | Desviacion estandar maxima de dias entre compras |
| `min_purchase_days_filter` | int | 10 | Dias minimos de compra (filtro de patron) |
| `min_products_filter` | int | 5 | Productos distintos minimos |
| `daily_output_file` | str | `daily.parquet` | Salida cliente-dia |
| `recommendation_output_file` | str | `filtered_agg_sales_for_rec.parquet` | Salida para recomendadores |

### `featurize`

Ingenieria de features para el modelo predictivo.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `future_days_offset` | int | 5 | Dias futuros a generar en el calendario |
| `quincena_days` | list | `[28,29,30,31,1,2,13,14,15,16]` | Dias de quincena (nomina) |
| `recency_fillna_days` | int | 365 | Valor para llenar `days_since_last` cuando no hay compra previa |
| `rolling_windows` | list | `[3, 7, 30]` | Ventanas para conteos rolling (`cnt_Xd`) |
| `monetary_windows` | list | `[7, 30]` | Ventanas para promedios monetarios (`avg_amount_Xd`, `avg_skus_Xd`) |
| `output_file` | str | `calendar_features.parquet` | Archivo de salida con features |

### `train`

Configuracion del modelo predictivo.

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `split_days_validation` | int | 30 | Dias para split temporal (train/valid) |
| `features` | list | ver abajo | Lista de features del modelo |
| `target` | str | `purchased` | Variable objetivo |
| `model_type` | str | `hist_gradient_boosting` | Tipo de modelo (`logistic_regression` o `hist_gradient_boosting`) |

Features: `dow`, `dom`, `month`, `is_weekend`, `is_quincena`, `days_since_last`, `cnt_3d`, `cnt_7d`, `cnt_30d`, `avg_amount_7d`, `avg_amount_30d`, `avg_skus_7d`, `avg_skus_30d`.

#### `train.logistic_regression`

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `solver` | `saga` | Algoritmo de optimizacion |
| `max_iter` | 100 | Iteraciones maximas |
| `tol` | 0.001 | Tolerancia de convergencia |
| `C` | 1.0 | Regularizacion inversa |
| `class_weight` | `balanced` | Peso de clases (compensa desbalance) |

#### `train.hist_gradient_boosting`

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `max_iter` | 200 | Arboles maximos |
| `max_depth` | 6 | Profundidad maxima por arbol |
| `learning_rate` | 0.1 | Tasa de aprendizaje |
| `min_samples_leaf` | 50 | Muestras minimas por hoja |
| `l2_regularization` | 0.1 | Regularizacion L2 |

### `model`

Rutas de modelos serializados.

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `model_dir` | `models` | Directorio de modelos |
| `model_name` | `model.joblib` | Modelo base |
| `calibrated_model_name` | `calibrated_model.joblib` | Modelo calibrado |

### `evaluate`

Configuracion de calibracion y evaluacion.

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `calibration_method` | `sigmoid` | Metodo de calibracion (`sigmoid` o `isotonic`) |
| `calibration_bins` | 10 | Bins para curva de calibracion |
| `evaluation_threshold` | 0.50 | Umbral de decision para metricas |
| `shap_sample_size` | 1000 | Muestras para SHAP (interpretabilidad) |

### `backtesting`

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `backtest_days` | 7 | Dias recientes a evaluar contra datos reales de la API |

### `recommendations_item_item`

Recomendador colaborativo item-item.

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `num_recommendations` | 5 | Recomendaciones por cliente |
| `excluded_product_descriptions` | (ver params.yaml) | Productos excluidos (bolsas, menu empleados) |
| `output_folder` | `recommendations` | Carpeta de salida (relativa a processed/) |
| `model_folder` | `recommendation_model_item_item` | Carpeta de artefactos del modelo |
| `transaction_data_processed_file` | `filtered_agg_sales_for_rec.parquet` | Datos de entrada |
| `similarity_matrix_file` | `item_similarity.npz` | Matriz de similitud |
| `mappings_file` | `mappings.pkl` | Mapeos usuario-item |
| `user_item_matrix_file` | `sparse_user_item.npz` | Matriz sparse |
| `precomputed_recs_file` | `precomputed_item_item_recs.parquet` | Recomendaciones precalculadas |

### `recommendations_clustering`

Segmentacion de clientes y recomendaciones por segmento.

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `months_recent_activity` | 1 | Meses recientes para filtrar clientes activos |
| `min_purchase_days_pattern` | 10 | Dias minimos de compra |
| `max_median_days_pattern` | 35 | Mediana maxima entre compras |
| `max_std_days_pattern` | 3 | Desviacion estandar maxima |
| `n_clusters` | 5 | Numero de clusters (5 perfiles de negocio) |
| `k_min` | 4 | k minimo para auto-seleccion (si `n_clusters` es null) |
| `k_max` | 7 | k maximo para auto-seleccion |
| `features_for_clustering` | `[tickets_per_month, ticket_median, months_active_ratio]` | Features de segmentacion |
| `excluded_product_descriptions` | (ver params.yaml) | Productos excluidos |
| `output_folder` | `recommendations` | Carpeta de salida |
| `precomputed_cluster_recs_file` | `precomputed_cluster_recs.parquet` | Recomendaciones por segmento |

### `recommendation_settings`

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `top_product_percentile` | 0.75 | Fraccion de productos a recomendar por historial reciente |

### `reports`

Rutas de reportes y graficos.

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `reports_dir` | `reports` | Directorio de reportes |
| `metrics_file` | `metrics.json` | Metricas de evaluacion |
| `base_class_report_file` | `classification_report_base.txt` | Reporte modelo base |
| `calibrated_class_report_file` | `classification_report_calibrated.txt` | Reporte modelo calibrado |
| `plots_dir` | `plots` | Subcarpeta de graficos |
| `calibration_plot` | `calibration_curve.png` | Curva de calibracion |
| `importance_plot` | `feature_importance.png` | Importancia de features |
| `shap_summary_plot` | `shap_summary.png` | Resumen SHAP |
| `backtesting_metrics_file` | `backtesting_metrics.csv` | Metricas de backtesting |

---

## dvc.yaml

Define el pipeline DVC de 6 etapas secuenciales.

### Etapas

#### `load`
```yaml
cmd: uv run python src/load_data.py --config params.yaml
```
- **Deps**: `src/load_data.py`, `params.yaml`
- **Outs**: `data/processed/initial_sales_clean.parquet` (persist)
- Descarga ventas de la API y actualiza el historico local.

#### `preprocess`
```yaml
cmd: uv run python src/preprocess.py --config params.yaml
```
- **Deps**: `src/preprocess.py`, `initial_sales_clean.parquet`, `params.yaml`
- **Outs**: `data/processed/daily.parquet` (persist)
- Filtra clientes, aplica patrones de regularidad, genera datasets.

#### `featurize`
```yaml
cmd: uv run python src/featurize.py --config params.yaml
```
- **Deps**: `src/featurize.py`, `daily.parquet`
- **Params**: `data.base_path`, `data.processed_folder`, `preprocess.daily_output_file`, `featurize.*`
- **Outs**: `data/processed/calendar_features.parquet` (persist)
- Genera calendario completo con features temporales, recencia y rolling.

#### `train`
```yaml
cmd: uv run python src/train.py --config params.yaml
```
- **Deps**: `src/train.py`, `calendar_features.parquet`
- **Params**: `base.random_state`, `data.*`, `featurize.output_file`, `train.*`, `model.*`
- **Outs**: `models/model.joblib` (persist)
- Entrena el modelo de clasificacion (logistic o gradient boosting).

#### `evaluate`
```yaml
cmd: uv run python src/evaluate.py --config params.yaml
```
- **Deps**: `src/evaluate.py`, `model.joblib`, `calendar_features.parquet`
- **Params**: `base.random_state`, `data.*`, `train.*`, `featurize.output_file`, `model.*`, `evaluate.*`, `reports.*`
- **Outs**: `models/calibrated_model.joblib` (persist), reportes de clasificacion
- **Metrics**: `reports/metrics.json`
- **Plots**: calibracion, importancia de features, SHAP
- Calibra el modelo y genera todos los reportes de evaluacion.

#### `backtest`
```yaml
cmd: uv run python src/backtest.py --config params.yaml
```
- **Deps**: `src/backtest.py`, `calibrated_model.joblib`, `calendar_features.parquet`
- **Params**: `data.*`, `model.*`, `evaluate.evaluation_threshold`, `backtesting.*`, `reports.*`, `train.features`
- **Outs**: `reports/backtesting_metrics.csv`
- Compara predicciones contra ventas reales de la API.

### Notas sobre DVC

- **`persist: true`**: Mantiene el archivo aunque DVC lo marque como desactualizado. Util para datos costosos de regenerar.
- **`cache: false`**: No cachea el archivo (metricas y plots). Permite ver diffs directamente en git.
- Los recomendadores y la segmentacion NO estan en el pipeline DVC — se ejecutan manualmente.
- Para re-ejecutar todo: `uv run dvc repro --force`.
