# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:26:38 2024

@author: KGP
"""

# SVR

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt # pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd # librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # Si pusiera [:, 1] en size de la tabla me daría (10,) porque lo consideraría un vector
y = dataset.iloc[:, 2].values
# iloc sirve para localizar por posición las variables, en este caso independientes
# hemos indicado entre los cochetes, coge todas las filas [:(todas las filas), :-1(todas las columnas excepto la última]
# .values significa que quiero sacar solo los valores del dataframe no las posiciones

"""
# Dividir el dataset en conjunto de entrenamiento y conjunto de testing 
from sklearn.model_selection import train_test_split #En esta ocasión al haber sólo 10 datos no hacemos trainning.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) # random_state podría coger cualquier número, es el número para poder reproducir el algoritmo

"""
# Escalado de variables. Siguiente código COMENTADO porque se usa mucho pero no siempre
from sklearn.preprocessing import StandardScaler # Utilizarlo para saber que valores debe escalar apropiadamente y luego hacer el cambio
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1, 1)) # hacemos transform sin "fit" para que haga la transformación con los datos del transform de entrenamiento


# Ajustar la regresión con el dataset. 
from sklearn.svm import SVR
regression = SVR(kernel = "rbf") # kernels en página sklearn, podemos elegir varios tipos
regression.fit(x, y)

# Predicción de nuestro modelos con SVR
pred = np.array([[6.5]])
y_pred = regression.predict(sc_x.transform(pred))
y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))

# Visualización de los resultados con SVR
    # podríamos aplicar el grid para suavizar la línea
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y.reshape(-1,1)), color = "red")
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regression.predict(x_grid).reshape(-1,1)), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

