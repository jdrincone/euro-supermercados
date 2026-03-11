# Metodologia de Ciencia de Datos

## 1. Problema de Negocio

Euro Supermercados (~20 tiendas en Colombia) quiere:
- **Predecir** que clientes compraran cada dia para lanzar campanas dirigidas.
- **Recomendar** productos relevantes a cada cliente.
- **Segmentar** clientes para disenar estrategias de marketing diferenciadas.

---

## 2. Datos

### Fuente
API interna de Euro Supermercados (`back-middleware.eurosupermercados.com`). Descarga ventas diarias con detalle de linea de factura.

### Esquema de ventas
| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `date_sale` | datetime | Fecha de la venta |
| `id_client` | string | Cedula del cliente |
| `product` | string | Codigo unico del producto |
| `invoice_value_with_discount_and_without_iva` | float | Valor de la linea (COP, sin IVA, con descuento) |
| `amount` | float | Cantidad (unidades o kg) |

### Volumen aproximado
- ~13M filas de ventas (6 meses)
- ~120k clientes unicos
- ~26k productos
- ~12k tickets/dia promedio

### Ventana temporal
Se mantienen los ultimos 6 meses de datos (`months_to_fetch: 6` en params.yaml). Al cargar datos nuevos de la API, se recorta automaticamente.

---

## 3. Validacion de Clientes

### Problema
No todos los IDs en los datos corresponden a personas naturales predecibles. Existen:
- IDs de empresas (NIT)
- IDs de prueba del sistema
- IDs genericos o erroneos

### Solucion: `client_filters.validate_client_ids()`

Se centralizo un filtro de cedula colombiana (CC) aplicado en tres puntos del pipeline:

| Criterio | Razon |
|----------|-------|
| Solo digitos | Excluye pasaportes alfanumericos, NITs con guion |
| No empieza en 0 | Formato CC colombiana |
| 6-10 digitos de longitud | Rango valido de CC |
| No esta en blacklist | Excluye IDs genericos (111111, 999999, 12345, etc.) |
| No es digito repetido | Excluye IDs como 7777777 |

---

## 4. Segmentacion de Clientes (Clustering)

### Objetivo
Identificar 5 perfiles de negocio:

| Perfil | Tickets/mes | Ticket (COP) | Meses activos | Estrategia |
|--------|-------------|--------------|---------------|------------|
| **Ballena** | >8 | >$200k | Alto | Son mini-mercados. Programa VIP, descuentos por volumen |
| **Cotidiano** | 2-8 | $50-150k | Alto | Familia que merca. Cross-sell, subir ticket |
| **Mensual** | ~1-2 | Medio | Moderado | Compra puntual. Incentivar segunda visita |
| **Hormiga** | Alto | <$30k | Alto | Visitas frecuentes, bajo monto. Subir ticket por visita |
| **Esporadico** | Bajo | Variable | Bajo | Irregular. Campanas de reactivacion |

### Features

Solo 3 features que separan directamente los perfiles:

| Feature | Formula | Que separa |
|---------|---------|-----------|
| `tickets_per_month` | total_tickets / meses_activos | Ballena/Hormiga (alto) vs Mensual (bajo) |
| `ticket_median` | mediana del valor de ticket diario | Ballena (alto COP) vs Hormiga (bajo COP) |
| `months_active_ratio` | meses_con_compra / meses_totales | Esporadico (bajo) vs los demas |

### Algoritmo

1. **Preparacion**: log-transform en `ticket_median` (distribucion sesgada), StandardScaler.
2. **K-Means** con k=5 (un cluster por perfil).
3. **Seleccion de k**: Silhouette score como metrica (configurable: `k_min`, `k_max`).
4. **Etiquetado automatico** (`label_clusters()`):
   - Menor `months_active_ratio` → Esporadico
   - Mayor `ticket_median` → Ballena
   - Mayor frecuencia + menor ticket → Hormiga
   - Restantes por `tickets_per_month` → Cotidiano vs Mensual

### Implicaciones para prediccion
- **Incluir**: Cotidiano, Mensual, Hormiga (patrones predecibles)
- **Excluir**: Esporadico (no hay patron), Ballena (dinamica B2B distinta)

---

## 5. Modelo Predictivo

### Tarea
Clasificacion binaria: para cada par (cliente, fecha), predecir si el cliente comprara ese dia.

- **Target**: `purchased` (1 si compro, 0 si no).
- **Granularidad**: cliente x dia.

### Ingenieria de Features

Todas las features usan `shift(1)` para evitar data leakage (la compra de hoy NO se usa para predecir hoy).

| Feature | Tipo | Descripcion |
|---------|------|-------------|
| `dow` | Temporal | Dia de la semana (0=Lun, 6=Dom) |
| `dom` | Temporal | Dia del mes (1-31) |
| `month` | Temporal | Mes (1-12) |
| `is_weekend` | Temporal | 1 si sabado/domingo |
| `is_quincena` | Temporal | 1 si es dia de nomina (1-2, 13-16, 28-31) |
| `days_since_last` | Recencia | Dias desde la ultima compra (max 365) |
| `cnt_3d` | Frecuencia | Compras en ultimos 3 dias |
| `cnt_7d` | Frecuencia | Compras en ultimos 7 dias |
| `cnt_30d` | Frecuencia | Compras en ultimos 30 dias |
| `avg_amount_7d` | Monetario | Monto promedio ultimos 7 dias |
| `avg_amount_30d` | Monetario | Monto promedio ultimos 30 dias |
| `avg_skus_7d` | Canasta | SKUs promedio ultimos 7 dias |
| `avg_skus_30d` | Canasta | SKUs promedio ultimos 30 dias |

### Calendario completo

`featurize.py` genera un calendario completo: todas las combinaciones (cliente, fecha) desde el primer dia hasta max_date + 5 dias. Esto permite predecir dias futuros.

### Modelos disponibles

Configurable via `train.model_type` en params.yaml:

| Modelo | Ventajas | Configuracion |
|--------|----------|---------------|
| `logistic_regression` | Rapido, interpretable, coeficientes claros | solver=saga, C=1.0, class_weight=balanced |
| `hist_gradient_boosting` | Mayor capacidad, maneja no-linealidades | max_iter=200, max_depth=6, lr=0.1 |

### Split temporal

NO se usa random split. Se hace split temporal estricto:

```
|---- TRAIN ----|---- VALID ----|
         train_end    max_purchase_date
         (max - 30d)
```

- **Train**: todas las fechas <= (max_date - 30 dias)
- **Valid**: fechas en (train_end, max_date]

### Calibracion

El modelo base no produce probabilidades bien calibradas. Se aplica calibracion sigmoide (`CalibratedClassifierCV`):

1. Split de validacion en dos mitades:
   - Primera mitad: conjunto de calibracion
   - Segunda mitad: conjunto de test (holdout puro)
2. Calibracion con `method='sigmoid'` sobre el conjunto de calibracion.
3. Evaluacion final sobre holdout.

Esto evita sobreajuste de la calibracion.

### Metricas

| Metrica | Que mide | Importancia |
|---------|----------|-------------|
| ROC-AUC | Capacidad de ranking | Metrica principal de discriminacion |
| Brier Score | Calidad de probabilidades | Menor es mejor |
| Precision @ 0.50 | De los predichos, cuantos compraron | Evita enviar campanas a quien no va a comprar |
| Recall @ 0.50 | De los compradores, cuantos predijimos | Cobertura |
| F0.5 @ 0.50 | Media ponderada (enfasis en precision) | Metrica de decision operativa |

Se usa F0.5 (no F1) porque el costo de un falso positivo (enviar campana innecesaria) es menor que perder un comprador, pero aun se prefiere precision.

---

## 6. Backtesting

Valida el modelo con datos reales de la API:

1. Descarga ventas reales de los ultimos N dias.
2. Predice clientes con prob >= umbral.
3. Compara predicciones vs ventas reales dia a dia.
4. Calcula Precision, Recall, F0.5 diarios.

Permite detectar degradacion del modelo antes de afectar campanas.

---

## 7. Motor de Recomendaciones

### 7.1 Por historial reciente (predict.py)

Para cada cliente predicho:
1. Filtra ventas del ultimo mes.
2. Calcula frecuencia de compra por producto.
3. Selecciona top 75% de productos (configurable).
4. Retorna lista ordenada por frecuencia.

### 7.2 Filtrado colaborativo item-item

Dos variantes:

**Global (train_recommender.py):**
1. Filtra al top 10% de productos por frecuencia global.
2. Construye matriz sparse usuario-item.
3. Calcula similitud coseno entre items.
4. Para cada cliente: suma similaridades ponderadas por frecuencia de compra.
5. Recomienda top 5 items no comprados.

**Por cliente (train_recommender_by_client.py):**
1. Selecciona top 5 productos por cliente.
2. Misma logica de similitud coseno.
3. Recomendaciones mas personalizadas.

### 7.3 Por segmento (train_recommender_by_clustering.py)

1. Segmenta clientes (5 perfiles, ver seccion 4).
2. Para cada cluster: identifica los top N productos mas populares.
3. N = promedio de productos distintos del cluster.
4. Excluye productos genericos (bolsas, menu empleados).

---

## 8. Decisiones de Diseno

### Por que logistic regression como opcion?
- Rapido de entrenar, pocas features.
- Coeficientes interpretables (cuanto pesa cada feature).
- Buena base comparativa.

### Por que hist_gradient_boosting como alternativa?
- Captura interacciones no-lineales (ej: quincena + dias_since_last).
- Maneja bien datos desbalanceados.
- Mejor rendimiento tipico en datos tabulares.

### Por que mediana y no media para ticket?
- La distribucion de tickets es muy sesgada (pocas ballenas, muchos tickets bajos).
- La mediana es robusta a outliers.

### Por que 3 features para clustering y no mas?
- Cada feature extra agrega ruido sin separar mejor los perfiles objetivo.
- tickets_per_month, ticket_median y months_active_ratio capturan directamente frecuencia, valor y regularidad.
- Menos features = clusters mas interpretables y estables.

### Por que excluir esporadicos y ballenas de prediccion?
- **Esporadicos**: Sin patron regular, el modelo no puede aprender cuando compraran.
- **Ballenas**: Son mini-mercados con dinamica B2B (compran por inventario, no por necesidad del hogar). Requieren un modelo diferente.
