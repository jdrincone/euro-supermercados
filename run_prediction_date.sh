#!/bin/bash

# Script para generar predicciones para un CONJUNTO FIJO de fechas
# y luego crear DOS archivos CSV:
# 1. Predicciones + Recomendaciones Item-Item
# 2. Predicciones + Recomendaciones por Clustering

# --- Configuración General ---
PYTHON_CMD="python3" # O "python" si es tu comando
# Asumimos que params.yaml está en el mismo directorio que este script .sh (raíz del proyecto)
CONFIG_FILE="params.yaml"

# --- Scripts de Python (rutas relativas al directorio raíz del proyecto) ---
PREDICT_SCRIPT="src/predict.py"
ADD_ITEM_ITEM_RECS_SCRIPT="src/get_recommendations.py"        # Script para item-item
ADD_CLUSTER_RECS_SCRIPT="src/add_cluster_recommendations.py" # Script para clustering

# --- Parámetros Específicos para este Script ---
TARGET_DATES="2025-05-13" # Fechas para las predicciones de compra
THRESHOLD="0.50"                     # Umbral de probabilidad de compra

# --- Nombres de Archivos y Carpetas (los scripts de Python leerán las rutas base de params.yaml) ---
# Los scripts de Python construirán las rutas completas usando la configuración de params.yaml
# Aquí solo definimos los nombres de archivo que se pasarán como argumentos --output

# Formatear las fechas para usarlas en los nombres de archivo
FORMATTED_DATES=$(echo "${TARGET_DATES}" | tr ' ' '_')

BASE_PREDICTIONS_FILENAME="predicciones_compra_${FORMATTED_DATES}.csv"
# El script predict.py lo guardará en la carpeta definida en params.yaml (o su default 'predictions')

OUTPUT_ITEM_ITEM_FILENAME="predicciones_con_recs_item_item_${FORMATTED_DATES}.csv"
# El script ADD_ITEM_ITEM_RECS_SCRIPT lo guardará en la carpeta definida en params.yaml (o su default 'recommendations')

OUTPUT_CLUSTER_FILENAME="predicciones_con_recs_cluster_${FORMATTED_DATES}.csv"
# El script ADD_CLUSTER_RECS_SCRIPT lo guardará en la carpeta definida en params.yaml (o su default 'recommendations')


echo "--- Iniciando proceso para fechas: ${TARGET_DATES} ---"
# Las carpetas de salida (predictions, recommendations) deben ser creadas por los scripts de Python
# si no existen, basándose en la configuración de params.yaml.

# --- Paso 0: Recordatorio sobre el pre-cálculo de los modelos de recomendación ---
echo "NOTA: Este script asume que los modelos y recomendaciones precalculadas para"
echo "      Item-Item y Clustering ya han sido generados y están actualizados."
echo "      - Item-Item: ejecutar el script de entrenamiento correspondiente (ej. src/train_recommender.py)."
echo "      - Clustering: ejecutar el script de pipeline de clustering (ej. src/recommender_pipeline.py o precompute_recommendation_model_clustering.sh)."
echo "      Ambos deben usar la configuración de '${CONFIG_FILE}' para que las rutas de salida coincidan."
echo ""


# --- Paso 1: Ejecutar predict.py para obtener probabilidades de compra ---
echo "Ejecutando predicción de compra ($PREDICT_SCRIPT)..."
"$PYTHON_CMD" "$PREDICT_SCRIPT" \
  --dates $TARGET_DATES \
  --threshold "$THRESHOLD" \
  --output "$BASE_PREDICTIONS_FILENAME" \
  --config "$CONFIG_FILE"

# Verificar éxito del Paso 1
if [ $? -ne 0 ]; then
    echo "Error: Falló la ejecución de $PREDICT_SCRIPT."
    exit 1
fi
# La verificación de existencia del archivo la hará el siguiente script al intentar leerlo.
echo "Predicciones de compra deberían estar en la carpeta definida por predict.py (usualmente 'predictions/')."
echo ""


# --- Paso 2a: Añadir recomendaciones Item-Item ---
echo "Añadiendo recomendaciones Item-Item ($ADD_ITEM_ITEM_RECS_SCRIPT)..."
# Este script debe leer BASE_PREDICTIONS_FILENAME de la carpeta de predicciones
# y guardar el resultado como OUTPUT_ITEM_ITEM_FILENAME en la carpeta de recomendaciones.
"$PYTHON_CMD" "$ADD_ITEM_ITEM_RECS_SCRIPT" \
  --input_file "$BASE_PREDICTIONS_FILENAME" \
  --output_file "$OUTPUT_ITEM_ITEM_FILENAME" \
  --config "$CONFIG_FILE"

# Verificar éxito del Paso 2a
if [ $? -ne 0 ]; then
    echo "Error: Falló la ejecución de $ADD_ITEM_ITEM_RECS_SCRIPT."
    # Considerar si quieres que el script falle aquí o continúe
    # exit 1
else
    echo "Recomendaciones Item-Item deberían estar en la carpeta definida por $ADD_ITEM_ITEM_RECS_SCRIPT (usualmente 'recommendations/')."
fi
echo ""


# --- Paso 2b: Añadir recomendaciones por Clustering ---
echo "Añadiendo recomendaciones por Clustering ($ADD_CLUSTER_RECS_SCRIPT)..."
# Este script debe leer BASE_PREDICTIONS_FILENAME de la carpeta de predicciones
# y guardar el resultado como OUTPUT_CLUSTER_FILENAME en la carpeta de recomendaciones.
"$PYTHON_CMD" "$ADD_CLUSTER_RECS_SCRIPT" \
  --input_predictions_file "$BASE_PREDICTIONS_FILENAME" \
  --output_file "$OUTPUT_CLUSTER_FILENAME" \
  --config "$CONFIG_FILE"

# Verificar éxito del Paso 2b
if [ $? -ne 0 ]; then
    echo "Error: Falló la ejecución de $ADD_CLUSTER_RECS_SCRIPT."
    # exit 1
else
    echo "Recomendaciones por Clustering deberían estar en la carpeta definida por $ADD_CLUSTER_RECS_SCRIPT (usualmente 'recommendations/')."
fi
echo ""

echo "--- Proceso Finalizado ---"
echo "Verifica los archivos en las carpetas 'predictions' y 'recommendations' (o las definidas en params.yaml)."
exit 0