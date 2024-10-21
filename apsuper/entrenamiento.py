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

data = pd.read_csv(os.path.join('data','breast-cancer.csv'))
print(data.head())

X = data.drop(columns=['id', 'diagnosis'])  # Excluir las columnas 'id' y 'diagnosis'
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convertir 'M' en 1 y 'B' en 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Capa de entrada
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo (40 épocas)
history = model.fit(X_train_scaled, y_train, epochs=40, batch_size=32, validation_data=(X_test_scaled, y_test))

model.save(os.path.join('models','mlp_model_super.h5'))


# Evaluar el modelo en el conjunto de prueba
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