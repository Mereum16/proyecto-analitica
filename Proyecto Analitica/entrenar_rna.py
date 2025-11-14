import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === 1. Cargar dataset ===
data = pd.read_excel("uploads/DEMALE-HSJM_2025_data (1).xlsx")

# === 2. Variable objetivo ===
target = "diagnosis"
X = data.drop(columns=[target])
y = data[target]

# === 3. División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Escalado ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Entrenar red neuronal ===
modelo_rna = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
modelo_rna.fit(X_train_scaled, y_train)

# === 6. Evaluar modelo ===
y_pred = modelo_rna.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred, average='weighted') * 100
rec = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print("\n MÉTRICAS DE LA RED NEURONAL (MLPClassifier)")
print(f"Accuracy: {acc:.2f}%")
print(f"Precision: {prec:.2f}%")
print(f"Recall: {rec:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# === 7. Matriz de confusión normalizada ===
cmatrix = confusion_matrix(y_test, y_pred, normalize='true')

os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cmatrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Matriz de Confusión Normalizada - Red Neuronal")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("reports/matriz_confusion_rna.png")
plt.close()

print("\n Matriz de confusión guardada en /reports/matriz_confusion_rna.png")

# === 8. Guardar modelo y escalador ===
os.makedirs("models", exist_ok=True)
with open("models/modelo_rna.pkl", "wb") as f:
    pickle.dump(modelo_rna, f)
with open("models/escalador_rna.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Modelo RNA guardado exitosamente en /models/")