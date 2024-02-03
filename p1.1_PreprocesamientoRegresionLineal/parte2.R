# Isaac Ulises Ascencio Padilla

source("R/Clases/Lib.Preprocess.R")
options(scipen = 999)
set.seed(1991)

#Importar datos
df.Wine <- read.csv(file = "Datasets/WineQT.csv", header = T, stringsAsFactors = F)

summary(df.Wine)

#NA
# df.Wine$alcohol <- ifelse(is.na(df.Wine$alcohol),
#                       ave(df.Wine$alcohol,
#                           FUN = function(x) mean(x, na.rm = T)
#                           ),
#                       df.Wine$alcohol)

#Factores
# df.Wine$pH <- factor(df.Wine$pH,
#                           levels = c("ejm","ejm","ejm"),
#                           labels = c(1,2,3))

#Escalado
df.Wine$pH
df.Wine[,9] <- scale(x = df.Wine[,9])

#Dividir
Split <- sample.split(Y = df.Wine$alcohol, SplitRatio = 0.8)
df.Wine.Train <- subset(df.Wine, Split == T)
df.Wine.Test <- subset(df.Wine, Split == F)

#Modelo Reg. lineal
mdl.Regresor <- lm(formula = alcohol ~ pH, data = df.Wine.Train)
summary(mdl.Regresor)

mdl.Predict <- predict(object = mdl.Regresor, newdata = df.Wine.Test)


plt.alcohol <- ggplot() + 
  theme_light() +
  ggtitle("Regresión: alcohol vs. pH") +
  xlab("pH") + 
  ylab("alcohol")

plt.alcohol.Data <- plt.alcohol + 
  geom_point(aes(x = df.Wine.Train$pH,
                 y = df.Wine.Train$alcohol), 
             colour = "blue2") +
  geom_line(aes(x = df.Wine.Test$pH,
                y = predict(mdl.Regresor, newdata = df.Wine.Test)),
            colour = "red3")

plt.alcohol.Data


#Validar la confianza de un modelo lineal
# Validación Cruzada (Cross-Validation): La validación cruzada es una técnica comúnmente
# utilizada para evaluar el rendimiento de un modelo. La validación cruzada divide
# el conjunto de datos en k subconjuntos (llamados "folds"), entrena el modelo en k-1
# de estos subconjuntos y lo evalúa en el subconjunto restante. Este proceso se repite
# k veces, utilizando un subconjunto diferente como conjunto de validación en cada iteración.
# Al final, se promedian los resultados de las k iteraciones para obtener una estimación
# general del rendimiento del modelo.

# # Instala el paquete si no lo tienes instalado
# install.packages("DAAG")

# # Carga la biblioteca
# library(DAAG)

# # Realiza la validación cruzada
# cv_result <- cv.lm(data = tu_conjunto_de_datos, form.lm = tu_formula)
# summary(cv_result)

# División de Datos en Entrenamiento y Prueba: Esta es la técnica más básica de validación.
# El conjunto de datos se divide en dos partes: un conjunto de entrenamiento y un conjunto
# de prueba. El modelo se entrena en el conjunto de entrenamiento y se evalúa en el conjunto
# de prueba. Esta técnica proporciona una estimación de cómo se desempeñará el modelo en datos
# no vistos.

# Estadísticas de Bondad de Ajuste
# summary(modelo)

# Estas estadísticas, como el coeficiente de determinación R^2, proporcionan información
# sobre la proporción de la variabilidad en la variable dependiente que es explicada por
# el modelo.