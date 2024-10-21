import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Cargar el dataset y los modelos
data = pd.read_csv(os.path.join('data','breast-cancer.csv'))
k_cluster=2
# Preprocesamiento: eliminar las columnas 'id' y 'diagnosis'
X = data.drop(columns=['id', 'diagnosis'])

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cargar el modelo K-Means previamente entrenado
kmeans = KMeans(n_clusters=k_cluster, random_state=42)
kmeans.fit(X_scaled)

# Asignar cada muestra a un cluster
data['cluster'] = kmeans.labels_

# Visualización de los clusters (usamos las dos primeras características para graficar)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['cluster'], cmap='viridis')
plt.title('Clustering con K-Means')
plt.xlabel('Característica 1 (normalizada)')
plt.ylabel('Característica 2 (normalizada)')
plt.show()