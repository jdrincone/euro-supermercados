# Sistema de Recomendación y Predicción de Clientes

Este proyecto implementa un sistema de recomendación y predicción de compras utilizando técnicas de Machine Learning.
El sistema combina un modelo predictivo para identificar clientes con alta probabilidad de
compra dada una fecha del futuro, con un sistema de recomendación basado en filtrado colaborativo item-item.



## 🚀 Características Principales

- **Predicción de Compra**: Modelo de clasificación que predice la probabilidad de compra de un cliente
- **Sistema de Recomendación**: Filtrado colaborativo item-item para recomendar productos
- **Backtesting**: Evaluación del modelo en períodos históricos
- **Generación de Predicciones**: Capacidad de generar predicciones bajo demanda

## 📁 Estructura del Proyecto

```
.
├── data/                      # Datos crudos y procesados
├── models/                    # Modelos entrenados
├── reports/                   # Métricas y visualizaciones
├── src/                       # Código fuente
│   ├── train.py              # Entrenamiento del modelo predictivo
│   ├── predict.py            # Generación de predicciones
│   ├── train_recommender.py  # Entrenamiento del recomendador
│   ├── get_recommendations.py # Obtención de recomendaciones
│   ├── evaluate.py           # Evaluación de modelos
│   ├── backtest.py           # Backtesting
│   ├── preprocess.py         # Preprocesamiento de datos
│   ├── featurize.py          # Generación de features
│   └── load_data.py          # Carga de datos
├── params.yaml               # Configuración del pipeline
├── dvc.yaml                  # Definición del pipeline DVC
└── requirements.txt          # Dependencias del proyecto
```

## 🛠️ Configuración del Entorno

### Prerrequisitos

- Python 3.11+
- Git
- DVC

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd recomendador_clientes
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

1. **Preprocesamiento**: Limpieza y transformación de datos
2. **Generación de Features**: Creación de características para el modelo
3. **Entrenamiento**: Entrenamiento del modelo predictivo
4. **Evaluación**: Validación del modelo
5. **Backtesting**: Prueba del modelo en datos históricos

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

El sistema de recomendación opera en dos fases:

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

