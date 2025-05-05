#!/bin/bash

# Script para generar predicciones para un CONJUNTO FIJO de fechas
# y luego añadir las recomendaciones al mismo archivo.

# --- Configuración ---
PYTHON_CMD="python3" # O "python" si es tu comando
PREDICT_SCRIPT="src/predict.py"
RECOMMEND_SCRIPT="src/get_recommendations.py"
PREDICTIONS_FOLDER="predictions" # Carpeta donde predict.py guarda por defecto
CONFIG_FILE="params.yaml"       # Asumiendo que los scripts usan esto por defecto

# --- Parámetros específicos ---
# Define aquí la lista de fechas separadas por espacio DENTRO de las comillas
TARGET_DATES="2025-05-01" # <--- Variable CORRECTA
THRESHOLD="0.55" # Umbral

# Nombre de archivo de salida más descriptivo
OUTPUT_FILENAME="predicciones.csv"
FULL_OUTPUT_FILEPATH="${PREDICTIONS_FOLDER}/${OUTPUT_FILENAME}"

# CORREGIDO: Usa la variable correcta TARGET_DATES
echo "--- Iniciando proceso para fechas: ${TARGET_DATES} ---"
mkdir -p "$PREDICTIONS_FOLDER"

# --- Paso 1: Ejecutar predict.py ---
echo "Ejecutando predicción ($PREDICT_SCRIPT)..."
# CORREGIDO: Usa la variable correcta TARGET_DATES
"$PYTHON_CMD" "$PREDICT_SCRIPT" \
  --dates $TARGET_DATES \
  --threshold "$THRESHOLD" \
  --output "$OUTPUT_FILENAME" \
  --config "$CONFIG_FILE"

# Verificar éxito del Paso 1
if [ $? -ne 0 ]; then
    echo "Error: Falló la ejecución de $PREDICT_SCRIPT."
    exit 1
fi
if [ ! -f "$FULL_OUTPUT_FILEPATH" ]; then
    echo "Error: El archivo de predicciones ($FULL_OUTPUT_FILEPATH) no fue creado por $PREDICT_SCRIPT."
    exit 1
fi
echo "Predicciones guardadas en: $FULL_OUTPUT_FILEPATH"

# --- Paso 2: Ejecutar get_recommendations.py ---
# Usa la RUTA COMPLETA como entrada y salida para enriquecer el archivo
echo "Añadiendo recomendaciones ($RECOMMEND_SCRIPT)..."
"$PYTHON_CMD" "$RECOMMEND_SCRIPT" \
  --input_file "$FULL_OUTPUT_FILEPATH" \
  --output_file "$FULL_OUTPUT_FILEPATH" \
  --config "$CONFIG_FILE"

# Verificar éxito del Paso 2
if [ $? -ne 0 ]; then
    echo "Error: Falló la ejecución de $RECOMMEND_SCRIPT."
    exit 1
fi

echo "Recomendaciones añadidas. Archivo final: $FULL_OUTPUT_FILEPATH"
echo "--- Proceso Finalizado ---"
exit 0