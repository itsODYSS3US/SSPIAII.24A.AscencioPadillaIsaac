# Isaac Ulises Ascencio Padilla

# Librerias
import random as rd
import math

# Dataset
dataset = [
    {"entrada":[0,0,1], "salida":0},
    {"entrada":[1,1,1], "salida":1},
    {"entrada":[1,0,1], "salida":1},
    {"entrada":[0,1,1], "salida":0}
]

dataset

class ANN():
    def __init__(self):
        rd.seed(4214)
        self.__pesos = [
            rd.uniform(-1, 1),
            rd.uniform(-1, 1),
            rd.uniform(-1, 1)
        ]
    
    #Núcleo - Core 
    def __nucleo(self, entradas):
        suma = 0
        for i, entradas in enumerate(entradas):
            suma += self.__pesos[i] * entradas
        return suma
    
    # Función de activación - sigmoide
    def __factivacion(self, score):
        return 1 / (1 + math.exp(-score))
    
    def __relu(self,x):
        return max(0, x)

    def __tanh(self,x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    # Predicción - Salida
    def predict(self, datos):
        score = self.__nucleo(datos)
        y_pred = self.__relu(score) # Cambiar la función de activación
        return y_pred

    # Función de coste 
    def __coste(self, y_pred, y_actual):
        return((y_pred - y_actual)**2)/2
    
    # Obtener pesos
    def getPesos(self):
        return self.__pesos
    
    
    # Training
    def training(self, datos, epochs):
        for epoch in range(epochs):
            for obs in datos:
                y_pred = self.predict(obs["entrada"])
                coste = self.__coste(y_pred, obs["salida"])

                error = obs["salida"] - y_pred

                # Ajustar pesos
                for index in range(len(self.__pesos)):
                    entrada = obs["entrada"][index]
                    adj = entrada * coste * error
                    print("Pesos adj",index,"-[",epoch,"]: ", adj)
                    self.__pesos[index] += adj
    
# Modelo
neurona = ANN()
print("Pesos iniciales: ", neurona.getPesos())

# Entrenamiento
neurona.training(dataset, 10) # Ajustar el número de epochs

# Predicción
prediccion = neurona.predict(dataset[0]["entrada"])


# Nuevas observaciones
nuevas_observaciones = [
    {"entrada": [0, 1, 0]},
    {"entrada": [1, 0, 0]},
    {"entrada": [0, 0, 0]},
]

# Predicciones
predicciones = []
for obs in nuevas_observaciones:
    prediccion = neurona.predict(obs["entrada"])
    predicciones.append(round(prediccion))  # Redondear a 0 o 1

# Matriz de confusión (asumiendo que las salidas reales son [1, 0, 0])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix([1, 1, 0], predicciones)
print(cm)



# Qué función de activación es la aducuada y por qué?
# Sigmoide suele ser una buena opción para problemas de clasificación binaria, ya que normaliza las salidas
# en un rango de entre 0 y 1, para este caso, considero que relu es una mejor opción ya que es más simple, 
# y el entramiento es más rápido, aunque no influye mucho aquí por el tamaño del dataset y al ver los ajustes
# de pesos relu presenta algunos valores en 0, esto significa que apaga las neuronas con entradas negativas,
# por lo que relu puede ser más adecuado para este caso, en el caso de tanh con su rango de -1 a 1 puede ser 
# menos intuitivo para la interpretación de los resultados para este tipo de problema de clasificación binaria.

# Cómo determinaste que era la mejor opción de entrenamiento?
# Con el número de 10 epochs los ajustes del modelo ya son muy pequeños, por lo que el modelos ya se está acercando
# a un mínimo en la función de pérdida, y no creo que aumentar el número de epochs pueda resultar en mejoras notables
# y al tratarse de un datasets pequeño, un número grande de epochs podría llevar un sobreajuste
