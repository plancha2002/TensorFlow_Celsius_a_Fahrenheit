import tensorflow as tf
import numpy as np
import pandas as pd
datos = pd.read_csv("celsius_a_fahrenheit.csv")
celsius = np.array(datos['Celsius'])
fahrenheit = np.array(datos['Fahrenheit'])
Linear_layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([Linear_layer])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(celsius, fahrenheit, epochs=500, verbose=False)
prediccion = np.array([1])
print(model.predict([prediccion]))
