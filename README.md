# Motor de recomendación: Predicción de compras y recomendación de productos

Este proyecto implementa un sistema de recomendación y predicción de compras utilizando técnicas de Machine Learning.
El sistema combina un modelo predictivo para identificar clientes con alta probabilidad de
compra dada una fecha del futuro, con un sistema de recomendación basado en filtrado colaborativo item-item.

## 🎯 Objetivos
- **Predecir**, Dada un fecha, calcular la probabilidad que un cliente realice una compra.  
- **Recomendar** los productos más adecuados a los clientes con alta probabilidad de compra.
        La recomendación se basa en sistema de filtrado colaborativo ítem-ítem

---
### Criterios de Filtrado
1. **Historial de Clientes**
   - Para clientes con compras en los últimos 3 meses (`last_month_with_sale`), se toma su historial de compras de los últimos
     6 meses (`months_to_fetch`)


2. **Patrones de Compra**
   - Mínimo de compra configurable (`min_purchase_count`)
   - Mediana de días entre compras menor a (`max_median_days_between`)
   - Desviación estándar de días entre compras menor a (`max_std_days_between`)

3. **Actividad General**
   - Máximo de días entre compra y compra (`min_purchase_days_filter`)
   - Mínimo de productos únicos comprados (`min_products_filter`)

## 🤖 Modelo Predictivo

### Feature Engineer

A partir del calendario completo **cliente‑fecha** se crean dos grupos de variables explicativas:

| Grupo | Variable | Descripción |
|-------|----------|-------------|
| **Temporales** | `dow` | Día de la semana |
| | `dom` | Día del mes |
| | `month` | Mes |
| | `is_weekend` | Indicador de fin de semana |
| | `is_quincena` | Indicador de quincena (28, 29, 30, 31, 1, 2, 13, 14, 15, 16 de cada mes) |
| | `days_since_last` | Días transcurridos desde la última compra |
| **Ventanas móviles (*lagged counts*)** | `cnt_1d` | Compras del cliente en el **día previo** |
| | `cnt_3d` | Compras acumuladas en los **3 días previos** |
| | `cnt_7d` | Compras acumuladas en la **última semana** |
| | `cnt_15d` | Compras acumuladas en los **últimos 15 días** |
| | `cnt_30d` | Compras acumuladas en el **último mes** |

Los conteos por ventana se calculan con un *rolling window* desplazado una fila para evitar fuga de información hacia el futuro:

### Características
- **Tipo**: Regresión Logística con calibración
- **Preprocesamiento**: Escalado estándar (sin centrado)
- **Características Principales**:
  - Historial de compras
  - Patrones temporales
  - Comportamiento de compra

### Pipeline de Entrenamiento
1. **Preprocesamiento**
   - Limpieza de datos
   - Filtrado de clientes según criterios
   - Patrones de compra
   - Agregación diaria de ventas
  

2. **Generación de Features**
   - Características de calendario

3. **Entrenamiento**
   - División train/validation basada en fechas
   - Entrenamiento con Regresión Logística
   - Calibración del modelo

4. **Evaluación**
   - Métricas de clasificación
   - Curva de calibración
   - Importancia de features
   - Análisis SHAP

## 🎯 Sistema de Recomendación

### Filtrado Colaborativo Item-Item
- **Alcance**: Últimos 3 de compras
- **Filtrado de Productos**:
  - Exclusión de productos específicos (ej: bolsas, transporte)
  - a. Selección basada en frecuencia de compra global se toma
       el 10% de los productos top comprados en los últimos 3 meses.
  - b. Selección basada en el top 5 de productos por cliente.

### Proceso de Recomendación
1. **Entrenamiento (Offline)**
   - Cálculo de matriz de similitud item-item
   - Pre-cálculo de recomendaciones para todos los clientes
   - Almacenamiento de mapeos y matrices

2. **Generación (Online)**
   - Búsqueda rápida de recomendaciones pre-calculadas
   - Filtrado por clientes con alta probabilidad de compra
   - Unión con información de productos


### Backtesting
- Evaluación diaria en período histórico
- Métricas por fecha
- Umbral de evaluación configurable

## 🛠️ Configuración del Entorno

### Prerrequisitos
- Python 3.11+
- Git
- DVC

### Instalación
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/jdrincone/euro.git
   cd euro
   ```

2. **Crear y activar entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   # venv\Scripts\activate    # Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar DVC (En caso de almacenar artefactos en la nube)**
   ```bash
   pip install dvc
   dvc remote add -d myremote s3://your-bucket/dvc-store
   dvc pull
   ```

## 🔄 Pipeline de Machine Learning

El pipeline completo para el modelo de probabilidad:
```bash
dvc repro
```

### Etapas del Pipeline
1. **load**: Carga y limpieza inicial de datos
2. **preprocess**: Filtrado y agregación de ventas
3. **featurize**: Generación de características
4. **train**: Entrenamiento del modelo
5. **evaluate**: Evaluación y calibración
6. **backtest**: Prueba en datos históricos

## 📊 Generación de Predicciones

Para generar predicciones de compra para una o varias fechas:

```bash
python3 src/predict.py --dates 2025-05-05 2025-05-06
```

Opciones adicionales:
- `--threshold`: Umbral de probabilidad (default: 0.5)
- `--output`: Nombre del archivo de salida
- `--config`: Ruta alternativa a params.yaml

### Ejemplo salida
Pensado durante unos segundos


| date       | client     | prob   | name                       | email                                                                                                     | phone      | telephone  |
| ---------- | ---------- | ------ | -------------------------- | --------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| 2025-05-01 | 64585834   | 0.8091 | GIMENEZ AVILA BLANCA ELENA | [elenablanca019@gmail.com](mailto:elenablanca019@gmail.com)                                               | 3127577573 | 3127577573 |
| 2025-05-01 | 1037669214 | 0.7904 | PACHECO  ALEJANDRA         | [facturacionelectronicapos@eurosupermercados.com](mailto:facturacionelectronicapos@eurosupermercados.com) | 3113825900 |            |
| 2025-05-01 | 1235245925 | 0.7904 | GISEL FERNANDEZ            | [gisellecfr@gmail.com](mailto:gisellecfr@gmail.com)                                                       | 3013475654 | 0          |


## 🎯 Sistema de Recomendación

### 1. Entrenamiento del Recomendador
#### En caso de predecir basado en el 10% de productos top
```bash
python3 src/train_recommender.py --config params.yaml
```

#### En caso de predecir basado en el top 5 de productos por cliente
```bash
python3 src/train_recommender_by_client.py --config params.yaml
```

### 2. Generación de Recomendaciones
```bash
python src/get_recommendations.py \
  --input_file predictions/predicciones_hoy.csv \
  --output_file recommendations/recomendaciones_para_hoy.csv
```
Pensado durante un par de segundos


| date       | client     | prob   | name                  | email                                                   | phone           | telephone | recommended\_product | recommendation\_score | recommendation\_rank | description                             | brand   | category         |
| ---------- | ---------- | ------ | --------------------- | ------------------------------------------------------- | --------------- | --------- | -------------------- | --------------------- | -------------------- | --------------------------------------- | ------- | ---------------- |
| 2025-05-01 | 1000291241 | 0.6036 | MENDOZA YUCELIS MARIA | [yuce31072002@gmail.com](mailto:yuce31072002@gmail.com) | 3114173080.0000 | 0.0000    | 113835               | 3.6345                | 1                    | CEREAL FLIPS DULCE LECHE BOLSA  x 120GR | FLIPS   | CEREALES         |
| 2025-05-01 | 1000291241 | 0.6036 | MENDOZA YUCELIS MARIA | [yuce31072002@gmail.com](mailto:yuce31072002@gmail.com) | 3114173080.0000 | 0.0000    | 74162                | 3.3973                | 2                    | CAFE GOURMET EUROMAX  x  500 GR         | EUROMAX | CAFE             |
| 2025-05-01 | 1000291241 | 0.6036 | MENDOZA YUCELIS MARIA | [yuce31072002@gmail.com](mailto:yuce31072002@gmail.com) | 3114173080.0000 | 0.0000    | 82462                | 2.8573                | 3                    | BUNUELO EURO 55 gr                      | EUROMAX | PANADERIA FRESCA |

##  Generación de predicción y recomendación usar sh
```bash
sh run_prediction_date.sh
```
ajustar dentro del sh los parámetros:

```bash
# --- Parámetros específicos ---
# Define aquí la lista de fechas separadas por espacio DENTRO de las comillas
TARGET_DATES="2025-05-01" # <--- Variable CORRECTA
THRESHOLD="0.55" # Umbral

# Nombre de archivo de salida más descriptivo
OUTPUT_FILENAME="prediccions_with_recommendation.csv"
FULL_OUTPUT_FILEPATH="${PREDICTIONS_FOLDER}/${OUTPUT_FILENAME}"
```

## 📈 Resultados y Métricas

Los resultados del pipeline se almacenan en:

- **Modelos**:
  - `models/model.joblib`: Modelo base
  - `models/calibrated_model.joblib`: Modelo calibrado

- **Reportes**:
  - `reports/metrics.json`: Métricas principales
  - `reports/classification_report_*.txt`: Reportes de clasificación
  - `reports/backtesting_metrics.csv`: Métricas de backtesting

- **Visualizaciones**:
  - `reports/plots/calibration_curve.png`
  - `reports/plots/feature_importance.png`
  - `reports/plots/shap_summary.png`

