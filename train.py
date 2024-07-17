import tensorflow as tf
import numpy as np
import os

# Generar datos de entrada aleatorios (por ejemplo, 1000 ejemplos con 10 características cada uno)
entrada = np.random.rand(1000, 10)  # 1000 ejemplos con 10 características cada uno
resultados = np.random.rand(1000) * 100  # 1000 resultados esperados aleatorios, escalados

# Definir la arquitectura de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, input_shape=(entrada.shape[1],)),  # Especificar input_shape en la primera capa
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
modelo.fit(entrada, resultados, epochs=100, verbose=0)

# Guardar el modelo y los pesos entrenados
modelo.save('modelo_musica.h5')  # Guarda el modelo en formato .h5
modelo.save_weights('pesos_musica.weights.h5')  # Guarda los pesos en formato .weights.h5

# Mensaje de confirmación
print("Entrenamiento completado con éxito. Modelos guardados correctamente.")

# Verificar la existencia de los archivos guardados
if os.path.exists('modelo_musica.h5') and os.path.exists('pesos_musica.weights.h5'):
    print("Archivos guardados encontrados en el directorio.")
else:
    print("No se encontraron los archivos guardados.")
