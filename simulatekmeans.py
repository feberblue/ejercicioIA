import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Simular nuevas características de un paciente basado en las mismas características del dataset original
# Simulamos 30 características que representen valores típicos de un paciente
nuevos_pacientes = np.array([
    [10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1, 10, 5, 10, 6, 8, 4, 7, 3, 9, 1],
    [7, 3, 5, 4, 6, 2, 4, 2, 5, 2, 7, 3, 5, 4, 6, 2, 4, 2, 5, 2, 7, 3, 5, 4, 6, 2, 4, 2, 5, 2],
    [30, 20, 151, 3000, 4, 5, 6, 7, 8, 9, 10, 20, 21, 0.9, 100, 23, 212, 45, 1, 1, 320, 3400, 1500, 2300, 45, 240, 1, 23, 45, 1 ],
    [20.57,	17.77, 132.9, 1326,	0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225,	0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902],
    [9.504,	12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076, 0.1815, 0.06905, 0.2773, 0.9768, 1.909, 15.7, 0.009606,	0.01432, 0.01985, 0.01421, 0.02027, 0.002968, 10.23, 15.66, 65.13, 314.9, 0.1324, 0.1148, 0.08867, 0.06227, 0.245, 0.07773]
])

K_clusters=4
# Cargar el dataset original para obtener el escalador y los parámetros de clustering
data = pd.read_csv(os.path.join('data','breast-cancer.csv'))

# Preprocesar los datos originales
X = data.drop(columns=['id', 'diagnosis'])  # Eliminar 'id' y 'diagnosis'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalizar las características

# Definir y entrenar el modelo K-Means con los datos originales
kmeans = KMeans(n_clusters=K_clusters, random_state=42)
kmeans.fit(X_scaled)

# Normalizar las características de los nuevos pacientes con el mismo escalador
nuevos_pacientes_scaled = scaler.transform(nuevos_pacientes)

# Usar el modelo K-Means para predecir a qué cluster pertenecen los nuevos pacientes
predicciones = kmeans.predict(nuevos_pacientes_scaled)

# Mostrar las predicciones
for i, prediccion in enumerate(predicciones):
    resultado = 'Posible cáncer' if prediccion == 1 else 'Probablemente no cáncer'
    print(f"Paciente {i+1}: {resultado}")

# Visualización de los nuevos pacientes en los clusters existentes
# Graficamos las dos primeras características de los pacientes simulados y los datos originales
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', label='Datos originales')
plt.scatter(nuevos_pacientes_scaled[:, 0], nuevos_pacientes_scaled[:, 1], color='red', marker='o', label='Nuevos pacientes')
plt.title('Clustering de Nuevos Pacientes con K-Means')
plt.xlabel('Característica 1 (normalizada)')
plt.ylabel('Característica 2 (normalizada)')
plt.legend()
plt.show()