import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


# Deshabilitar las optimizaciones de oneDNN para evitar advertencias de rendimiento
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Cargar el modelo entrenado
# Se pueden cambiar entre el modelo mpl_model.h5 y mlp_model_super.h5
model = tf.keras.models.load_model(os.path.join('models','mlp_model_super.h5'))

# Cargar el dataset original para obtener el escalador
data = pd.read_csv(os.path.join('data','breast-cancer.csv'))

# Preprocesamiento: Asegurarnos de que solo usemos 30 características
# 'diagnosis' es la etiqueta que queremos predecir, y 'id' no es relevante
X = data.drop(columns=['id', 'diagnosis'])  # Eliminamos 'id' y 'diagnosis'
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convertimos 'M' en 1 y 'B' en 0

# Verificar que las características de entrada sean 30
print(f"Número de características: {X.shape[1]}")  # Debe imprimir 30

# Normalización de las características de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ajustar el escalador con los datos originales

# Nuevas muestras con exactamente 30 características (valores ficticios)
nuevas_muestras = np.array([
                            [10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1],
                            [30, 20, 151, 3000, 4, 5, 6, 7, 8, 9, 10, 20, 21, 0.9, 100, 23, 212, 45, 1, 1, 320, 3400, 1500, 2300, 45, 240, 1, 23, 45, 1 ],
                            [20.57,	17.77, 132.9, 1326,	0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225,	0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902],
                            [9.504,	12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076, 0.1815, 0.06905, 0.2773, 0.9768, 1.909, 15.7, 0.009606,	0.01432, 0.01985, 0.01421, 0.02027, 0.002968, 10.23, 15.66, 65.13, 314.9, 0.1324, 0.1148, 0.08867, 0.06227, 0.245, 0.07773]])  

if nuevas_muestras.shape[1] != 30:
    raise ValueError("Las nuevas muestras deben tener exactamente 30 características.")

# Escalar las nuevas muestras con el mismo escalador usado en el entrenamiento
nuevas_muestras_scaled = scaler.transform(nuevas_muestras)

# Realizar la predicción con el modelo
predicciones = model.predict(nuevas_muestras_scaled)

# Convertir la predicción en un valor legible (benigno o maligno) segun el número de muestras
for prediccion in predicciones: 
    resultado = 'Maligno' if prediccion > 0.5 else 'Benigno'
    print(f"El tumor es: {resultado} con una predicción de =>{prediccion}")
