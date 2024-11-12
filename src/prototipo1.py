import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos simulados para fines de prototipo
data_terridata = {
    "municipio": ["Bucaramanga", "Floridablanca", "Girón", "Piedecuesta"],
    "poblacion_total": [581130, 267849, 206932, 163622],
    "poblacion_migrante": [50000, 20000, 15000, 10000],
    "nivel_pobreza": [0.25, 0.20, 0.30, 0.22],
    "acceso_salud": [0.80, 0.85, 0.75, 0.78],
}

data_sivigila = {
    "municipio": ["Bucaramanga", "Floridablanca", "Girón", "Piedecuesta"],
    "casos_sarampion": [10, 5, 8, 3],
    "fecha_reporte": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
}

# Convertir a DataFrame
df_terridata = pd.DataFrame(data_terridata)
df_sivigila = pd.DataFrame(data_sivigila)

# Combinar conjuntos de datos
df_combined = pd.merge(df_terridata, df_sivigila, on="municipio")

# Crear una puntuación de riesgo basada en población migrante y nivel de pobreza
df_combined["risk_score"] = (
    df_combined["poblacion_migrante"] * df_combined["nivel_pobreza"]
)

# Definir características y variable objetivo
X = df_combined[
    [
        "poblacion_total",
        "poblacion_migrante",
        "nivel_pobreza",
        "acceso_salud",
        "risk_score",
    ]
]
y = df_combined["casos_sarampion"]

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar un modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# Función para predecir el riesgo de casos de sarampión en un nuevo municipio
def predict_risk(poblacion_total, poblacion_migrante, nivel_pobreza, acceso_salud):
    risk_score = poblacion_migrante * nivel_pobreza
    features = np.array(
        [[poblacion_total, poblacion_migrante, nivel_pobreza, acceso_salud, risk_score]]
    )
    prediction = model.predict(features)
    return prediction[0]


# Ejemplo de predicción para un nuevo municipio
new_municipio_prediction = predict_risk(200000, 30000, 0.28, 0.80)
print(f"Predicted cases of sarampión: {new_municipio_prediction}")
