import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# === Configuración visual ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# === Cargar dataset ===
data = pd.read_excel("uploads/DEMALE-HSJM_2025_data (1).xlsx")

# === Columna objetivo ===
target_col = "diagnosis"

# === Codificar etiquetas si son texto ===
le = LabelEncoder()
data[target_col] = le.fit_transform(data[target_col])

# === Variables predictoras ===
X = data.drop(columns=[target_col])
y = data[target_col]

# === Entrenar modelo Random Forest ===
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# === Importancia de variables ===
importances = rf.feature_importances_
feat_importances = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": importances
}).sort_values(by="Importancia", ascending=False)

# === Mostrar ranking en consola ===
print(" Top 15 Variables Más Importantes en el Diagnóstico:")
print(feat_importances.head(15).to_string(index=False))

# === Gráfico profesional ===
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importancia", 
    y="Variable", 
    data=feat_importances.head(15),
    edgecolor="black"
)
plt.title("Importancia de Variables – Proyecto Analítica", fontsize=14, color="#004080", pad=15)
plt.xlabel("Nivel de Importancia", fontsize=12)
plt.ylabel("Variable Clínica", fontsize=12)
plt.tight_layout()

# === Guardar gráfico ===
plt.savefig("uploads/Importancia_Variables_Proyecto_Analitica.png", dpi=300)
plt.show()

print("\n Gráfico guardado como 'uploads/Importancia_Variables_Proyecto_Analitica.png'")