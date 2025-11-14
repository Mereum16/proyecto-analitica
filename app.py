# --- imports ---
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Etiquetas de las clases (asegúrate de que coincidan con tu codificación)
CLASES = ["Dengue", "Malaria", "Leptospirosis"]


app = Flask(__name__)
app.secret_key = "superseguro"

# --- CONFIG: rutas a modelos/escaladores ---
MODEL_DIR = "models"
LOG_MODEL_BAL = os.path.join(MODEL_DIR, "modelo_logistica_balanceado.pkl")
RNA_MODEL = os.path.join(MODEL_DIR, "modelo_rna.pkl")
RNA_SCALER = os.path.join(MODEL_DIR, "escalador_rna.pkl")

# --- lee dataset base para inferir rangos ---
DATASET_PATH = os.path.join("uploads", "DEMALE-HSJM_2025_data (1).xlsx")
df = pd.read_excel(DATASET_PATH)

if "diagnosis" in df.columns:
    feature_cols = [c for c in df.columns if c != "diagnosis"]
else:
    feature_cols = df.columns.tolist()

# --- nombres bonitos ---
DISPLAY_NAME = {
    "male": "Sexo masculino (1=Sí, 0=No)",
    "female": "Sexo femenino (1=Sí, 0=No)",
    "age": "Edad (años)",
    "urban_origin": "Origen urbano (1=Sí, 0=No)",
    "rural_origin": "Origen rural (1=Sí, 0=No)",
    "homemaker": "Ama de casa (1=Sí, 0=No)",
    "student": "Estudiante (1=Sí, 0=No)",
    "professional": "Profesional (1=Sí, 0=No)",
    "merchant": "Comerciante (1=Sí, 0=No)",
    "agriculture_livestock": "Agricultura/Ganadería (1=Sí, 0=No)",
    "various_jobs": "Varios oficios (1=Sí, 0=No)",
    "unemployed": "Desempleado (1=Sí, 0=No)",
    "hospitalization_days": "Días de hospitalización",
    "body_temperature": "Temperatura corporal (°C)",
    "fever": "Fiebre (1=Sí, 0=No)",
    "headache": "Cefalea (1=Sí, 0=No)",
    "dizziness": "Mareo (1=Sí, 0=No)",
    "loss_of_appetite": "Pérdida de apetito (1=Sí, 0=No)",
    "weakness": "Debilidad (1=Sí, 0=No)",
    "myalgias": "Mialgias (1=Sí, 0=No)",
    "arthralgias": "Artralgias (1=Sí, 0=No)",
    "eye_pain": "Dolor ocular (1=Sí, 0=No)",
    "hemorrhages": "Hemorragias (1=Sí, 0=No)",
    "vomiting": "Vómito (1=Sí, 0=No)",
    "abdominal_pain": "Dolor abdominal (1=Sí, 0=No)",
    "chills": "Escalofríos (1=Sí, 0=No)",
    "hemoptysis": "Hemoptisis (1=Sí, 0=No)",
    "edema": "Edema (1=Sí, 0=No)",
    "jaundice": "Ictericia (1=Sí, 0=No)",
    "bruises": "Moretones (1=Sí, 0=No)",
    "petechiae": "Petequias (1=Sí, 0=No)",
    "rash": "Exantema (1=Sí, 0=No)",
    "diarrhea": "Diarrea (1=Sí, 0=No)",
    "respiratory_difficulty": "Dificultad respiratoria (1=Sí, 0=No)",
    "itching": "Prurito (1=Sí, 0=No)",
    "hematocrit": "Hematocrito (%)",
    "hemoglobin": "Hemoglobina (g/dL)",
    "red_blood_cells": "Glóbulos rojos (mm³)",
    "white_blood_cells": "Glóbulos blancos (mm³)",
    "neutrophils": "Neutrófilos (%)",
    "eosinophils": "Eosinófilos (%)",
    "basophils": "Basófilos (%)",
    "monocytes": "Monocitos (%)",
    "lymphocytes": "Linfocitos (%)",
    "platelets": "Plaquetas (µL)",
    "AST (SGOT)": "AST (SGOT) (U/L)",
    "ALT (SGPT)": "ALT (SGPT) (U/L)",
    "ALP (alkaline_phosphatase)": "Fosfatasa alcalina (U/L)",
    "total_bilirubin": "Bilirrubina total (mg/dL)",
    "direct_bilirubin": "Bilirrubina directa (mg/dL)",
    "indirect_bilirubin": "Bilirrubina indirecta (mg/dL)",
    "total_proteins": "Proteínas totales (g/dL)",
    "albumin": "Albúmina (g/dL)",
    "creatinine": "Creatinina (mg/dL)",
    "urea": "Urea (mg/dL)",
}

# --- tipos y rangos ---
variables_info = {}
for col in feature_cols:
    series = df[col].dropna()
    unique_vals = set(series.unique().tolist())
    is_binary = unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0})
    if is_binary:
        vtype = "binary"
        vmin, vmax, step = 0, 1, 1
    else:
        vtype = "number"
        vmin = float(np.nanmin(series))
        vmax = float(np.nanmax(series))
        step = 1 if pd.api.types.is_integer_dtype(series) else 0.01

    variables_info[col] = {
        "display": DISPLAY_NAME.get(col, col),
        "type": vtype,
        "min": round(vmin, 2) if vtype == "number" else vmin,
        "max": round(vmax, 2) if vtype == "number" else vmax,
        "step": step
    }

# --- unificamos sexo ---
if "male" in variables_info: del variables_info["male"]
if "female" in variables_info: del variables_info["female"]
FEATURE_ORDER = [c for c in df.columns if c != "diagnosis"]

# --- utilitario: construir fila ---
def build_input_row(form_dict):
    sexo = int(form_dict.get("sexo"))
    male = 1 if sexo == 1 else 0
    female = 1 - male

    row = []
    for col in FEATURE_ORDER:
        if col == "male":
            row.append(male)
        elif col == "female":
            row.append(female)
        else:
            val = form_dict.get(col)
            if val is None or val == "":
                row.append(np.nan)
                continue
            if col in variables_info and variables_info[col]["type"] == "binary":
                row.append(int(val))
            else:
                row.append(float(val))
    return np.array(row).reshape(1, -1)

# --- cargar modelos ---
def load_pair(model_path, scaler_path):
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

# Cargar modelo balanceado (pipeline completo)
import pickle

if os.path.exists(LOG_MODEL_BAL):
    with open(LOG_MODEL_BAL, "rb") as f:
        log_model = pickle.load(f)
else:
    log_model = None

rna_model, rna_scaler = load_pair(RNA_MODEL, RNA_SCALER)


# --- INDIVIDUAL ---
# --- INDIVIDUAL ---
@app.route("/individual", methods=["GET", "POST"])
def individual():
    if request.method == "GET":
        return render_template("individual.html", variables_info=variables_info)

    # --- Recolección del formulario ---
    form = request.form.to_dict()

    # --- Validación y recorte de valores según rango ---
    for k, meta in variables_info.items():
        if k in form and meta["type"] == "number" and form[k] != "":
            x = float(form[k])
            x = max(meta["min"], min(meta["max"], x))
            form[k] = str(x)

    # --- Construir entrada ---
    X = build_input_row(form)

    # --- Obtener el modelo seleccionado (default = logistica) ---
    modelo = request.form.get("modelo", "logistica")

    # --- Seleccionar modelo ---
    if modelo == "logistica":
        model = log_model
    else:
        model = rna_model

    if model is None:
        flash("❌ No se encontró el modelo seleccionado.", "danger")
        return redirect(url_for("individual"))

    # --- Predicción (el pipeline incluye el escalado) ---
    y_pred = int(model.predict(X)[0])

    # --- Probabilidades ---
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        prob_max = float(np.max(probas))
        clase_pred = CLASES[y_pred] if y_pred < len(CLASES) else str(y_pred)
        prob_txt = f"{prob_max * 100:.2f}%"
    else:
        clase_pred = CLASES[y_pred] if y_pred < len(CLASES) else str(y_pred)
        prob_txt = "—"

    # --- Mensaje final ---
    mensaje = f"🧠 Diagnóstico probable: {clase_pred}"

    resultado = {
        "mensaje": mensaje,
        "prob": prob_txt
    }

    # --- Renderizar ---
    return render_template("individual.html", variables_info=variables_info, resultado=resultado)



# --- LOTES ---
# --- LOTES ---
@app.route("/lotes", methods=["GET", "POST"])
def lotes():
    tabla = None
    metricas = None
    cm_base64 = None
    model_name = None

    if request.method == "POST":
        file = request.files.get("dataset")
        modelo = request.form.get("modelo", "logistica")  # por defecto usa el modelo balanceado

        if not file:
            flash("⚠️ Debes subir un archivo válido (.xlsx o .csv).", "danger")
            return redirect(url_for("lotes"))

        # --- Leer dataset ---
        try:
            data = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
        except Exception as e:
            flash(f"❌ Error al leer el archivo: {e}", "danger")
            return redirect(url_for("lotes"))

        # --- Validar columnas ---
        if "diagnosis" not in data.columns:
            flash("❌ El dataset no contiene la columna 'diagnosis'.", "danger")
            return redirect(url_for("lotes"))

        X = data.drop(columns=["diagnosis"])
        y_true = data["diagnosis"]

        # --- Seleccionar modelo ---
        if modelo == "logistica":
            model = log_model
            model_name = "Regresión Logística (Balanceada)"
        else:
            model = rna_model
            model_name = "Red Neuronal"

        if model is None:
            flash("❌ No se encontró el modelo seleccionado.", "danger")
            return redirect(url_for("lotes"))

        # --- Predicción (el pipeline ya incluye escalado/SMOTE) ---
        y_pred = model.predict(X)

        # --- Calcular métricas ---
        acc = accuracy_score(y_true, y_pred) * 100
        unique_classes = np.unique(y_true)
        avg_type = "weighted" if len(unique_classes) > 2 else "binary"

        prec = precision_score(y_true, y_pred, average=avg_type) * 100
        rec = recall_score(y_true, y_pred, average=avg_type) * 100
        f1 = f1_score(y_true, y_pred, average=avg_type) * 100

        metricas = {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2)
        }

        # --- Matriz de confusión (normalizada para ver proporciones) ---
        # --- Rebalancear datos antes de evaluar ---
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y_true)

        # --- Predicciones sobre los datos re-muestreados ---
        y_pred_bal = model.predict(X_res)

        # --- Matriz de confusión con conteos enteros ---
        cm = confusion_matrix(y_res, y_pred_bal)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusión (Balanceada) - {model_name}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        cm_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)


        # --- Guardar resultados ---
        data["Predicción"] = y_pred
        global resultados_lotes
        resultados_lotes = data.copy()

        tabla = data.head(50).to_html(classes="table table-striped table-bordered", index=False)
        flash("✅ Predicción por lotes completada correctamente.", "success")

    return render_template("lotes.html", tabla=tabla, metricas=metricas, cm=cm_base64, model_name=model_name)


# --- DESCARGAR RESULTADOS ---
@app.route("/descargar_resultados")
def descargar_resultados():
    global resultados_lotes
    if resultados_lotes is None:
        flash("❌ No hay resultados para descargar.", "danger")
        return redirect(url_for("lotes"))
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        resultados_lotes.to_excel(writer, index=False, sheet_name="Resultados")
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="resultados_lotes.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- INDEX ---
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    # host 0.0.0.0 para que sea accesible externamente

    app.run(host="0.0.0.0", port=port)
