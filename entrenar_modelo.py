import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# === 1. Cargar dataset ===
data = pd.read_excel("uploads/DEMALE-HSJM_2025_data (1).xlsx")

# === 2. Variable objetivo ===
target = "diagnosis"
X = data.drop(columns=[target])
y = data[target]

# === 3. División de datos (train/test) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Pipeline con escalado + balanceo + modelo ===
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, solver='lbfgs'))
])

# === 5. Validación cruzada estratificada ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=False
)

print("\n MÉTRICAS PROMEDIO (Validación Cruzada 5-Fold)")
print(f"Accuracy: {cv_results['test_accuracy'].mean() * 100:.2f}%")
print(f"Precision: {cv_results['test_precision_weighted'].mean() * 100:.2f}%")
print(f"Recall: {cv_results['test_recall_weighted'].mean() * 100:.2f}%")
print(f"F1-Score: {cv_results['test_f1_weighted'].mean() * 100:.2f}%")

# === 6. Entrenamiento final en todo el conjunto de entrenamiento ===
pipeline.fit(X_train, y_train)

# === 7. Evaluar en el conjunto de prueba ===
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred, average='weighted') * 100
rec = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print("\n MÉTRICAS FINALES EN TEST")
print(f"Accuracy: {acc:.2f}%")
print(f"Precision: {prec:.2f}%")
print(f"Recall: {rec:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# === 8. Matriz de confusión normalizada ===
cmatrix = confusion_matrix(y_test, y_pred, normalize='true')

os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cmatrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Matriz de Confusión Normalizada - Regresión Logística Balanceada")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("reports/matriz_confusion_logistica_balanceada.png")
plt.close()

print("\n Matriz de confusión guardada en /reports/matriz_confusion_logistica_balanceada.png")

# === 9. Guardar modelo (pipeline completo) ===
os.makedirs("models", exist_ok=True)
with open("models/modelo_logistica_balanceado.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(" Modelo balanceado guardado exitosamente en /models/modelo_logistica_balanceado.pkl")
