import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Deshabilitar las optimizaciones de oneDNN para evitar advertencias de rendimiento
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

hidden_layer = 4
n_neurones_hidden_layer = 32
n_neurones_layer_initial = 64

data = pd.read_csv(os.path.join('data','breast-cancer.csv'))

X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  

# se dividen los datos para entrenamiento y para pruebas (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(n_neurones_layer_initial, activation='relu', input_shape=(X_train_scaled.shape[1],))) # capa de entrada con n Neuronas
for layern in range(hidden_layer):
    name_layer = f"layer{layern+1}"
    model.add(Dense(n_neurones_hidden_layer, activation='relu', name=name_layer)) # capa oculta

model.add(Dense(1, activation='sigmoid')) # Salida con funcion de activacion (salida binaria)

# Compilar el modelo tipo optimizacion adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo (40 épocas)
history = model.fit(X_train_scaled, y_train, epochs=40, batch_size=32, validation_data=(X_test_scaled, y_test))

# Se genera modelo respecto al numero de capas ocultas
name_model_layers=f"mlp_model_super_{hidden_layer}_layers.h5"
model.save(os.path.join('models',name_model_layers))

loss, accuracy = model.evaluate(X_test_scaled, y_test)

print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")

# Graficar la precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Precisión del modelo durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()