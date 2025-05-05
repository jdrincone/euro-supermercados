# Motor de recomendación: Predicción de compras y recomendación de productos

Este proyecto implementa un sistema de recomendación y predicción de compras utilizando técnicas de Machine Learning.
El sistema combina un modelo predictivo para identificar clientes con alta probabilidad de
compra dada una fecha del futuro, con un sistema de recomendación basado en filtrado colaborativo item-item.

## 🎯 Objetivos
- **Predecir**, Dada un fecha, calcular la probabilidad de realizar al menos una compra de un cliente.  
- **Recomendar** los productos más adecuados a los clientes con alta probabilidad de compra.
        La recomendación es en base al historico del cliente

---
### Criterios de Filtrado
1. **Historial de Clientes**
   - Para clientes con compras en los últimos 3 meses, se toma su historial de compras de los últimos
     6 meses (`months_to_fetch`)


2. **Patrones de Compra**
   - Mínimo de compra configurable (`min_purchase_count`)
   - Mediana de días entre compras menor a (`max_median_days_between`)
   - Desviación estándar de días entre compras menor a (`max_std_days_between`)

3. **Actividad General**
   - Mínimo de días de compra (`min_purchase_days_filter`)
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
   - Filtrado de clientes según criterios
   - Agregación diaria de ventas
   - Limpieza de datos

2. **Generación de Features**
   - Características de calendario
   - Patrones de compra
   - Métricas de comportamiento

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
- **Alcance**: Últimos 3 meses de datos
- **Filtrado de Productos**:
  - Exclusión de productos específicos (ej: bolsas, transporte)
  - Selección basada en frecuencia de compra (Q50-Q100)
  - Productos con descripción válida

### Proceso de Recomendación
1. **Entrenamiento (Offline)**
   - Cálculo de matriz de similitud item-item
   - Pre-cálculo de recomendaciones para todos los clientes
   - Almacenamiento de mapeos y matrices

2. **Generación (Online)**
   - Búsqueda rápida de recomendaciones pre-calculadas
   - Filtrado por clientes con alta probabilidad de compra
   - Unión con información de productos

## 📊 Métricas y Evaluación

### Modelo Predictivo
- **Métricas Principales**:
  - Precisión
  - Recall
  - F1-Score
  - Curva de calibración

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

4. **Configurar DVC**
   ```bash
   pip install dvc
   dvc remote add -d myremote s3://your-bucket/dvc-store
   dvc pull
   ```

## 🔄 Pipeline de Machine Learning

El pipeline completo se ejecuta con:
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
python src/predict.py --dates 2025-05-05 2025-05-06
```

Opciones adicionales:
- `--threshold`: Umbral de probabilidad (default: 0.5)
- `--output`: Nombre del archivo de salida
- `--config`: Ruta alternativa a params.yaml

## 🎯 Sistema de Recomendación

### 1. Entrenamiento del Recomendador
```bash
python src/train_recommender.py --config params.yaml
```

### 2. Generación de Recomendaciones
```bash
python src/get_recommendations.py \
  --input_file predictions/predicciones_hoy.csv \
  --output_file recommendations/recomendaciones_para_hoy.csv
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

