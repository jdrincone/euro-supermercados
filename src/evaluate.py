# src/evaluate.py
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse
import joblib
import json
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix,
    classification_report, precision_score, recall_score, fbeta_score
)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from datetime import timedelta

# Configurar matplotlib para backend no interactivo (útil para DVC)
import matplotlib
matplotlib.use('Agg')

def evaluate_model(config_path):
    """Evalúa, calibra y genera métricas/plots del modelo."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config['data']
    train_params = config['train']
    eval_params = config['evaluate']
    base_path = Path(data_params['base_path'])
    processed_path = base_path / data_params['processed_folder']
    model_path = Path(config['model']['model_dir'])
    reports_path = Path(config['reports']['reports_dir'])
    plots_path = reports_path / config['reports']['plots_dir']
    plots_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True) # Asegurar que reports_path existe

    # Cargar datos y modelo
    features_file = processed_path / config['featurize']['output_file']
    calendar = pd.read_parquet(features_file)
    model_file = model_path / config['model']['model_name']
    pipe = joblib.load(model_file)
    print(f"Modelo cargado desde: {model_file}")

    # Definir fechas de corte para validación (igual que en train.py)
    max_hist = calendar[calendar['purchased'] == 1]['date'].max()
    if pd.isna(max_hist):
        max_hist = calendar['date'].max() - timedelta(days=train_params['split_days_validation'])
    train_end_date = max_hist - timedelta(days=train_params['split_days_validation'])
    valid_end_date = max_hist

    valid_mask = (calendar["date"] > train_end_date) & (calendar["date"] <= valid_end_date)
    features = train_params['features']
    target = train_params['target']

    X_valid = calendar.loc[valid_mask, features]
    y_valid = calendar.loc[valid_mask, target]
    print(f"Datos de validación cargados. Filas: {len(X_valid)}")

    # --- Evaluación del Modelo Base (Sin Calibrar) ---
    print("Evaluando modelo base (sin calibrar)...")
    y_pred_base = pipe.predict(X_valid)
    y_prob_base = pipe.predict_proba(X_valid)[:, 1]

    roc_auc_base = roc_auc_score(y_valid, y_prob_base)
    brier_base = brier_score_loss(y_valid, y_prob_base)
    print(f"  ROC-AUC (Base): {roc_auc_base:.3f}")
    print(f"  Brier Score (Base): {brier_base:.3f}")
    # print("  Confusion Matrix (Base):\n", confusion_matrix(y_valid, y_pred_base)) # Comentado para no llenar stdout

    # ---- Guardar Reporte de Clasificación Base ----
    report_base_str = classification_report(y_valid, y_pred_base, digits=3)
    base_report_file = reports_path / config['reports']['base_class_report_file']
    with open(base_report_file, 'w') as f:
        f.write("Classification Report (Modelo Base - Sin Calibrar)\n")
        f.write("==================================================\n")
        f.write(report_base_str)
    print(f"  Reporte de clasificación base guardado en: {base_report_file}")
    # -------------------------------------------------

    # --- Calibración ---
    print(f"Calibrando modelo usando método: {eval_params['calibration_method']}")
    calibrator = CalibratedClassifierCV(
        pipe,
        method=eval_params['calibration_method'],
        cv='prefit' # Usar 'prefit' porque pipe ya está entrenado
    )
    # Capturar warnings durante fit si es necesario
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="The `cv='prefit'` option is deprecated")
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        calibrator.fit(X_valid, y_valid) # Calibrar sobre el set de validación

    calibrated_model_file = model_path / config['model']['calibrated_model_name']
    joblib.dump(calibrator, calibrated_model_file)
    print(f"Modelo calibrado guardado en: {calibrated_model_file}")

    # --- Evaluación del Modelo Calibrado ---
    print("Evaluando modelo calibrado...")
    y_prob_cal = calibrator.predict_proba(X_valid)[:, 1]
    # Usar el umbral definido en params.yaml para el reporte
    y_pred_cal_thresh = (y_prob_cal >= eval_params['evaluation_threshold']).astype(int)

    roc_auc_cal = roc_auc_score(y_valid, y_prob_cal)
    brier_cal = brier_score_loss(y_valid, y_prob_cal)
    precision_cal = precision_score(y_valid, y_pred_cal_thresh)
    recall_cal = recall_score(y_valid, y_pred_cal_thresh)
    f05_cal = fbeta_score(y_valid, y_pred_cal_thresh, beta=0.5)

    print(f"  ROC-AUC (Calibrado): {roc_auc_cal:.3f}")
    print(f"  Brier Score (Calibrado): {brier_cal:.3f}")
    print(f"  Evaluación con umbral = {eval_params['evaluation_threshold']}:")
    print(f"    Precision: {precision_cal:.3f}")
    print(f"    Recall: {recall_cal:.3f}")
    print(f"    F0.5 Score: {f05_cal:.3f}")
    # print("  Confusion Matrix (Calibrado):\n", confusion_matrix(y_valid, y_pred_cal_thresh)) # Comentado

    # ---- Guardar Reporte de Clasificación Calibrado ----
    report_cal_str = classification_report(y_valid, y_pred_cal_thresh, digits=3)
    calibrated_report_file = reports_path / config['reports']['calibrated_class_report_file']
    with open(calibrated_report_file, 'w') as f:
        f.write(f"Classification Report (Modelo Calibrado - Umbral {eval_params['evaluation_threshold']:.2f})\n")
        f.write("=========================================================\n")
        f.write(report_cal_str)
    print(f"  Reporte de clasificación calibrado guardado en: {calibrated_report_file}")
    # ----------------------------------------------------

    # --- Guardar Métricas ---
    metrics = {
        'roc_auc_base': roc_auc_base,
        'brier_base': brier_base,
        'roc_auc_calibrated': roc_auc_cal,
        'brier_calibrated': brier_cal,
        f'precision_at_{eval_params["evaluation_threshold"]}': precision_cal,
        f'recall_at_{eval_params["evaluation_threshold"]}': recall_cal,
        f'f0.5_at_{eval_params["evaluation_threshold"]}': f05_cal,
    }
    metrics_file = reports_path / config['reports']['metrics_file']
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {metrics_file}")

    # --- Plot de Calibración ---
    print("Generando plot de calibración...")
    fig_cal, ax_cal = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_estimator(
        pipe, X_valid, y_valid, n_bins=eval_params['calibration_bins'],
        strategy="uniform", name="Modelo Base", ax=ax_cal
    )
    CalibrationDisplay.from_estimator(
        calibrator, X_valid, y_valid, n_bins=eval_params['calibration_bins'],
        strategy="uniform", name="Modelo Calibrado", ax=ax_cal
    )
    ax_cal.set_title("Curva de Calibración")
    ax_cal.grid(alpha=0.3)
    plt.tight_layout()
    cal_plot_file = plots_path / config['reports']['calibration_plot']
    plt.savefig(cal_plot_file)
    plt.close(fig_cal)
    print(f"Plot de calibración guardado en: {cal_plot_file}")

    # --- Feature Importance (Coeficientes de Regresión Logística) ---
    print("Generando plot de Feature Importance...")
    try:
        # Acceder al estimador dentro del calibrador y luego al pipeline
        if hasattr(calibrator, 'base_estimator_'): # sklearn >= 1.4 approx
            pipeline_in_calibrator = calibrator.base_estimator_
        else: # Versiones anteriores
            pipeline_in_calibrator = calibrator.estimator

        scaler = pipeline_in_calibrator.named_steps['standardscaler']
        logreg = pipeline_in_calibrator.named_steps['logisticregression']

        # Obtener coeficientes y nombres de features
        coefficients = logreg.coef_[0]
        feature_names = features # Asumiendo que 'features' es la lista de nombres

        # Crear dataframe para plotear
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coefficients)})
        importance_df = importance_df.sort_values('importance', ascending=False).head(15) # Top 15

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(importance_df['feature'], importance_df['importance'])
        ax_imp.set_xlabel('Importancia (Valor Absoluto del Coeficiente)')
        ax_imp.set_title('Top 15 Feature Importances (Modelo Calibrado)')
        ax_imp.invert_yaxis()
        plt.tight_layout()
        imp_plot_file = plots_path / config['reports']['importance_plot']
        plt.savefig(imp_plot_file)
        plt.close(fig_imp)
        print(f"Plot de Feature Importance guardado en: {imp_plot_file}")

    except Exception as e:
        print(f"Error generando Feature Importance: {e}")


    # --- SHAP Values ---
    print("Calculando y generando plot SHAP...")
    try:
        # Necesitamos datos escalados para el LinearExplainer
        # Usar el mismo scaler que está dentro del pipeline calibrado
        X_valid_scaled = scaler.transform(X_valid)

        # Tomar una muestra para eficiencia si X_valid_scaled es muy grande
        sample_size = min(eval_params['shap_sample_size'], len(X_valid_scaled))
        if sample_size < len(X_valid_scaled):
            indices = np.random.choice(X_valid_scaled.shape[0], sample_size, replace=False)
            X_sample_scaled = X_valid_scaled[indices]
            print(f"Usando muestra de {sample_size} para SHAP.")
        else:
            X_sample_scaled = X_valid_scaled
            print(f"Usando {sample_size} datos completos para SHAP.")

        # Convertir a DataFrame para que SHAP use nombres de features
        X_sample_scaled_df = pd.DataFrame(X_sample_scaled, columns=feature_names)

        # Explicador para el modelo de regresión logística (la parte relevante del pipeline)
        explainer = shap.LinearExplainer(logreg, X_sample_scaled_df)
        shap_values = explainer.shap_values(X_sample_scaled_df)

        # Summary Plot (tipo barra)
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, X_sample_scaled_df, plot_type="bar", show=False)
        plt.title("Importancia de Features SHAP (Modelo Calibrado)")
        plt.tight_layout()
        shap_plot_file = plots_path / config['reports']['shap_summary_plot']
        plt.savefig(shap_plot_file)
        plt.close(fig_shap)
        print(f"Plot SHAP Summary guardado en: {shap_plot_file}")

    except Exception as e:
        print(f"Error calculando/generando SHAP plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml', help='Path to config file')
    args = parser.parse_args()
    evaluate_model(args.config)