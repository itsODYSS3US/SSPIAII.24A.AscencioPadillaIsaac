# Isaac Ulises Ascencio Padilla

import tensorflow as tf
# print(tf.__version__)

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

df_Bank = pd.read_csv('datasets/Bank.csv')

X = df_Bank.iloc[:,3:13].values
Y = df_Bank.iloc[:,13].values


# Preprocesamiento
# Dummies
X[:,1] = LabelEncoder().fit_transform(X[:,1])
X[:,2] = LabelEncoder().fit_transform(X[:,2])

one = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[1])], 
    remainder='passthrough'
    )

X = one.fit_transform(X)
X = X[:,1:]

# Escalado
X = StandardScaler().fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)


# Modelo ANN
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()

# Capa de entrada
ann.add(
    Dense(units = 6, kernel_initializer='uniform', input_dim = 11)
)

# Capas ocultas
ann.add(
    Dense(units = 2, activation="tanh", kernel_initializer='uniform')
)

ann.add(
    Dense(units = 2, activation="tanh", kernel_initializer='uniform')
)

# Capa de salida
ann.add(
    Dense(units = 1, activation='sigtmoid', kernel_initializer = "uniform")
)


# Entrenador
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

print(X_train.shape)

ann.fit(X_train, Y_train, epochs = 50, batch_size= 50)


# Guardar el modelo
from keras.models import load_model
ann.save('ann.h5')

modelo = load_model('ann.h5')


Y_pred = modelo.predict(X_test)
Y_pred = (Y_pred > 0.5)


# Visualizar la arquitectura de la red neuronal
from keras.utils import plot_model

plot_model(modelo,to_file='model.png',show_shapes=True,show_layer_activations=True,show_layer_names=True) 