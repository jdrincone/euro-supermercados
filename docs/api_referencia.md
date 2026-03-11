# Referencia de Modulos

Referencia tecnica de todas las funciones publicas en `src/`.

---

## config.py

Carga de configuracion YAML y resolucion de rutas.

### `load_config(path: str | Path = "params.yaml") -> dict`
Carga `params.yaml` y valida secciones requeridas (`data`, `base`).

### `processed_path(cfg) -> Path`
Retorna `data.base_path / data.processed_folder`.

### `model_dir(cfg) -> Path`
Retorna `model.model_dir`.

### `reports_dir(cfg) -> Path`
Retorna `reports.reports_dir`.

### `plots_dir(cfg) -> Path`
Retorna `reports_dir / reports.plots_dir`.

---

## api_client.py

Cliente HTTP para la API de Euro Supermercados. Credenciales via `API_USERNAME` / `API_PASSWORD` en `.env`.

### `get_auth_token(username=None, password=None) -> str`
Obtiene token de autenticacion. Lee de `.env` si no se pasan parametros.

### `fetch_sales(token, start_date, end_date) -> tuple[list[dict], list[str]]`
Descarga ventas dia a dia. Retorna `(registros_json, fechas_fallidas)`.

### `fetch_third_parties(token, documents, batch_size=10) -> list[dict]`
Consulta datos de contacto de clientes en lotes de `batch_size`.

---

## data_io.py

Operaciones de I/O centralizadas.

### `load_parquet(path, label="") -> pd.DataFrame`
Carga Parquet y loguea tamano. Lanza `FileNotFoundError` si no existe.

### `save_parquet(df, path, label="") -> None`
Guarda Parquet. Crea directorios padres si no existen.

### `load_product_catalog(path) -> pd.DataFrame`
Carga CSV de productos. Renombra `codigo_unico` a `product`, elimina duplicados. Retorna `[product, description]`.

### `load_calendar_features(cfg) -> pd.DataFrame`
Carga `calendar_features.parquet` con tipos normalizados (`date` datetime, `client` string).

### `load_calibrated_model(cfg) -> Any`
Carga `calibrated_model.joblib`. Lanza `FileNotFoundError` si no existe.

### `save_model(model, path) -> None`
Serializa modelo con joblib.

### `save_text_report(path, header, content) -> None`
Guarda reporte de texto con encabezado subrayado.

### `save_json(path, obj) -> None`
Guarda diccionario como JSON indentado.

---

## client_filters.py

Validacion de cedulas colombianas.

### `validate_client_ids(df, id_col="id_client", min_length=6, max_length=10) -> pd.DataFrame`
Filtra filas con IDs validos de persona natural:
- Solo digitos, no empieza en 0, longitud 6-10.
- No esta en blacklist (IDs genericos).
- No es digito repetido (ej: 7777777).

Retorna DataFrame filtrado con IDs limpiados (stripped).

---

## patterns.py

Analisis de patrones de compra y features de segmentacion.

### `filter_recent_clients(df, months, date_col="date_sale", client_col="id_client") -> pd.DataFrame`
Mantiene clientes con compras en los ultimos `months` meses. Usa max fecha del dataset como referencia.

### `compute_purchase_patterns(df, client_col, date_col, product_col, amount_col=None) -> pd.DataFrame`
Metricas por cliente: `purchase_days`, `median_days_between`, `std_days_between`, `product_distinct`, `pay_amount_mean`.

### `apply_pattern_filters(patterns, min_purchase_days, max_median_days, max_std_days, min_products=None) -> pd.Series`
Filtra clientes por criterios de regularidad. Retorna Serie con IDs validos.

### `compute_segmentation_features(df, client_col, date_col, product_col, amount_col) -> pd.DataFrame`
Features para segmentacion (5 perfiles):
- `tickets_per_month`: promedio de tickets por mes activo.
- `ticket_median`: mediana del valor del ticket (COP).
- `months_active_ratio`: fraccion de meses con al menos 1 compra.
- `products_per_visit`: productos distintos promedio por visita.
- `product_distinct`: total de productos unicos.
- `monetary_total`: gasto acumulado total (COP).

### `label_clusters(features, cluster_col="cluster") -> pd.DataFrame`
Asigna etiquetas de negocio a clusters segun centroides:
1. Menor `months_active_ratio` → esporadico
2. Mayor `ticket_median` → ballena
3. Mayor frecuencia + menor ticket → hormiga
4. Restantes → cotidiano / mensual

Agrega columnas `segment` y `segment_label`.

---

## collaborative.py

Filtrado colaborativo item-item.

### `create_sparse_matrix(df, client_col="client", product_col="product") -> tuple[csr_matrix | None, dict]`
Construye matriz sparse usuario-item. `mappings` contiene: `user_map`, `product_map`, `user_map_inv`. Retorna `(None, {})` si datos insuficientes.

### `compute_item_similarity(matrix) -> csr_matrix | None`
Similitud coseno item-item. Retorna `None` si < 2 productos.

### `recommend_for_client(client_idx, sparse_matrix, similarity_matrix, product_map, n_recs, return_scores=False) -> list`
Genera recomendaciones ponderando similaridad por frecuencia de compra. Retorna top N productos no comprados.

---

## load_data.py

Etapa DVC `load`. Sincroniza ventas locales con la API.

### `load_and_clean(config_path) -> None`
1. Carga historico local (Parquet).
2. Si esta al dia, no hace nada.
3. Descarga ventas nuevas de la API.
4. Limpia (normaliza fechas, valida IDs con `client_filters`).
5. Combina, deduplica, recorta ventana temporal.
6. Guarda Parquet actualizado.

---

## preprocess.py

Etapa DVC `preprocess`. Filtra clientes activos y genera datasets.

### `preprocess_sales(config_path) -> None`
1. Carga `initial_sales_clean.parquet`.
2. Valida IDs con `client_filters`.
3. Filtra clientes recientes.
4. Calcula patrones y aplica filtros de regularidad.
5. Genera `filtered_agg_sales_for_rec.parquet` (cliente-producto-dia).
6. Genera `daily.parquet` (cliente-dia).

---

## featurize.py

Etapa DVC `featurize`. Construye calendario completo con features.

### `create_features(config_path) -> None`
1. Carga `daily.parquet`.
2. Genera calendario completo (todos los clientes x todas las fechas + 5 dias futuros).
3. Agrega features temporales: `dow`, `dom`, `month`, `is_weekend`, `is_quincena`.
4. Agrega recencia: `days_since_last` (shift(1) para evitar leakage).
5. Agrega conteos rolling: `cnt_3d`, `cnt_7d`, `cnt_30d` (shift(1)).
6. Agrega features monetarias: `avg_amount_7d`, `avg_amount_30d`, `avg_skus_7d`, `avg_skus_30d` (shift(1)).

---

## train.py

Etapa DVC `train`. Entrena modelo de clasificacion.

### `train_model(config_path) -> None`
1. Carga `calendar_features.parquet`.
2. Split temporal (train: todo antes de max_date - 30d, valid: resto).
3. Construye pipeline segun `model_type`:
   - `logistic_regression`: StandardScaler + LogisticRegression
   - `hist_gradient_boosting`: HistGradientBoostingClassifier
4. Entrena y guarda `model.joblib`.

---

## evaluate.py

Etapa DVC `evaluate`. Calibra modelo y genera reportes.

### `evaluate_model(config_path) -> None`
1. Carga modelo base y calendar_features.
2. Split en 3: train (ya usado), calibracion, test (holdout).
3. Evalua modelo base en test.
4. Calibra con sigmoide en conjunto de calibracion.
5. Evalua calibrado en test.
6. Guarda `calibrated_model.joblib`.
7. Genera: metrics.json, classification reports, calibration curve, PR curve, feature importance, SHAP.

---

## backtest.py

Etapa DVC `backtest`. Compara predicciones vs ventas reales.

### `perform_backtesting(config_path) -> None`
1. Define rango de backtesting (ultimos N dias).
2. Descarga ventas reales de la API.
3. Predice clientes con prob >= umbral.
4. Calcula TP/FP/FN y metricas diarias.
5. Guarda `backtesting_metrics.csv`.

---

## predict.py

Genera predicciones operativas + recomendaciones.

### `main(config_path, prediction_dates_str, threshold_override=None, output_filename=..., recommendation_months=1) -> None`
1. Carga modelo calibrado y calendar_features.
2. Predice para fechas dadas.
3. Consulta datos de contacto via API.
4. Genera recomendaciones por historial reciente.
5. Guarda CSVs con predicciones + recomendaciones + contacto.

---

## train_recommender.py

Recomendador item-item global (top 10% productos).

### `train_and_save(config_path) -> None`
Filtra top 10% productos, construye similitud coseno, precalcula recomendaciones para todos los clientes.

---

## train_recommender_by_client.py

Recomendador item-item por cliente (top 5 productos/cliente).

### `train_and_save(config_path) -> None`
Selecciona top 5 productos por cliente, calcula similitud y genera recomendaciones personalizadas.

---

## train_recommender_by_clustering.py

Segmentacion + recomendaciones por segmento.

### `main_pipeline(config_path) -> None`
1. Carga ventas, valida IDs.
2. Filtra clientes recientes.
3. Calcula features de segmentacion (`compute_segmentation_features`).
4. Ejecuta K-Means (k=5 o auto-seleccion por silhouette).
5. Asigna etiquetas de negocio (`label_clusters`).
6. Genera recomendaciones por segmento.
7. Guarda `cluster_assignments.parquet` y `precomputed_cluster_recs.parquet`.

---

## get_recommendations.py

Une recomendaciones precalculadas con predicciones.

### `get_recommendations(input_file, output_file, config_path="params.yaml", include_clustering=False) -> None`
1. Carga CSV de predicciones.
2. Carga recomendaciones item-item.
3. Opcionalmente carga recomendaciones por clustering.
4. Hace left-join por `client`.
5. Guarda CSV combinado.

---

## utils.py

> **Deprecado.** Modulo de compatibilidad que re-exporta funciones con nombres antiguos. Importar directamente desde `config`, `api_client` o `data_io`.

| Alias antiguo | Funcion real |
|---------------|-------------|
| `read_yaml` | `config.load_config` |
| `obtener_token` | `api_client.get_auth_token` |
| `obtener_ventas` | `api_client.fetch_sales` |
| `obtener_terceros` | `api_client.fetch_third_parties` |
