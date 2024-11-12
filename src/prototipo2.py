import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests


# Paso 1: Recopilación de Datos
def fetch_data():
    # Ejemplo de cómo obtener datos de Terridata y Sivigila
    terridata_url = "URL_DE_TERRIDATA"
    sivigila_url = "URL_DE_SIVIGILA"

    terridata = pd.read_csv(terridata_url)
    sivigila = pd.read_csv(sivigila_url)

    return terridata, sivigila


# Paso 2: Combinación de Datos
def combine_data(terridata, sivigila):
    combined_data = pd.merge(terridata, sivigila, on="municipio")
    return combined_data


# Paso 3: Preprocesamiento de Datos
def preprocess_data(data):
    # Ejemplo de preprocesamiento
    data.fillna(0, inplace=True)
    return data


# Paso 4: Entrenamiento del Modelo Predictivo
def train_model(data):
    X = data.drop(columns=["caso_confirmado"])
    y = data["caso_confirmado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


# Paso 5: Generación de Alertas
def generate_alerts(model, new_data):
    predictions = model.predict(new_data)
    alerts = new_data[predictions == 1]
    return alerts


# Integración con la Plataforma Interactiva y la Aplicación Móvil
def integrate_with_platform(alerts):
    # Ejemplo de integración con la plataforma "Salud Conectada"
    for index, alert in alerts.iterrows():
        # Enviar alerta a la plataforma
        requests.post("URL_DE_LA_PLATAFORMA", data=alert.to_dict())


# Ejecución del Algoritmo
terridata, sivigila = fetch_data()
combined_data = combine_data(terridata, sivigila)
preprocessed_data = preprocess_data(combined_data)
model = train_model(preprocessed_data)

# Simulación de nuevos datos para generar alertas
new_data = pd.DataFrame(
    {
        "municipio": ["Municipio1", "Municipio2"],
        "porcentaje_migrante": [30, 45],
        "nivel_pobreza": [20, 35],
        "casos_reportados": [5, 10],
    }
)

alerts = generate_alerts(model, new_data)
integrate_with_platform(alerts)
