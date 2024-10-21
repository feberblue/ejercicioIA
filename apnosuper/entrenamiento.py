import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar el dataset
data = pd.read_csv(os.path.join('data','breast-cancer.csv'))

# Preprocesamiento: eliminar las columnas 'id' y 'diagnosis'
X = data.drop(columns=['id', 'diagnosis'])

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# calculamos el numero de clusters
def plot_elbow_method(scaled_data):
    inertia = [KMeans(n_clusters=k, random_state=0).fit(scaled_data).inertia_ for k in range(1, 11)]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, 'bo-', markersize=8, color='royalblue')
    plt.xlabel('Numero de cluster', fontsize=12)
    plt.ylabel('Inercia', fontsize=12)
    plt.title('Metodo del Codo Optimos=> k', fontsize=14)
    plt.grid(True)
    plt.show()

plot_elbow_method(X_scaled)
# Definir y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Guardar el modelo entrenado
if not os.path.exists('model_ns'):
    os.makedirs('model_ns')

# Guardar los centros del clustering
centers = kmeans.cluster_centers_
pd.DataFrame(centers).to_csv(os.path.join('model_ns','kmeans_centers_3k.csv'), index=False)

print("Entrenamiento completado. El modelo K-Means ha sido entrenado y los centros han sido guardados.")