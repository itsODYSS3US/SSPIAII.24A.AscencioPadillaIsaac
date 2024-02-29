# Isaac Ulises Ascencio Padilla

source("Lib.Preprocess.R")
getwd()
setwd("C:/Users/uliss/OneDrive/Documentos/UdG/SSP IA II/R/SSPIAII.24A.AscencioPadillaIsaac")

set.seed(1991)
options(scipen = 999)

#Datos
df.Salaries <- read.csv("Datasets/Position_Salaries.csv", header = T, stringsAsFactors = T)
df.Salaries$Position <- NULL

summary(df.Salaries)

mdl.Salaries <- rpart(Salary ~ ., df.Salaries, minsplit = 1)
summary(mdl.Salaries)

rpart.plot(mdl.Salaries)

x <- seq(min(df.Salaries[,1]) -1, max(df.Salaries[,1]) + 1, by = 0.1)
new.Data <- as.data.frame(x)
colnames(new.Data) <- "Level"

ggplot() + geom_point(aes(x = df.Salaries$Level, y = df.Salaries$Salary), colour ="darkred") +
    geom_line(aes(x = df.Salaries$Level, y = predict(mdl.Salaries, newdata = df.Salaries)), colour = "darkblue")+
    geom_line(aes(x = new.Data$Level, y = predict(mdl.Salaries, newdata = new.Data)), colour = "green")



# Parte 2: Modelos de clasificación con "Social Network Ads" y evaluación

# Cargar y explorar el dataset "Social Network Ads"
df.Social <- read.csv("Datasets/Social_Network_Ads.csv", header = T, stringsAsFactors = T)

df.Social$User.ID <- NULL
df.Social$Gender <- NULL

df.Social$Age <- scale(df.Social$Age)
df.Social$EstimatedSalary <- scale(df.Social$EstimatedSalary)


colnames(df.Social) <- c('Age','EstimatedSalary','Purchased')
df.Social$Age <- as.numeric(df.Social$Age)
df.Social$EstimatedSalary <- as.numeric(df.Social$EstimatedSalary)

# Split <- sample.split(df.Social$Purchased, SplitRatio = 0.8)
# df.Social.Train <- subset(df.Social, Split == T)
# df.Social.Test <- subset(df.Social, Split == F)

# Modelo de árboles de decisión (rpart)
mdl.rpart <- rpart(Purchased ~ ., data = df.Social, method = "class")

# Predicciones del modelo de árboles de decisión
predictions_tree <- predict(mdl.rpart, df.Social, type = "class")

# Matriz de confusión para el modelo de árboles de decisión

# Convertir la columna Purchased a factor
df.Social$Purchased <- as.factor(df.Social$Purchased)

# Convertir las predicciones a factor con los mismos niveles que Purchased
predictions_tree <- factor(predictions_tree, levels = levels(df.Social$Purchased))

# Ahora, puedes calcular la matriz de confusión sin error
confusion_matrix_tree <- confusionMatrix(data = predictions_tree, reference = df.Social$Purchased)
confusion_matrix_tree


# Modelo de bosques aleatorios (randomForest)
model_forest <- randomForest(Purchased ~ ., data = df.Social)

# Predicciones del modelo de bosques aleatorios
predictions_forest <- predict(model_forest, df.Social)

# Matriz de confusión para el modelo de bosques aleatorios
confusion_matrix_forest <- confusionMatrix(data = predictions_forest, 
                                           reference = df.Social$Purchased)
confusion_matrix_forest
# Comparación de la efectividad de los modelos
print("Matriz de confusión - Árbol de Decisión:")
print(confusion_matrix_tree)
print("Matriz de confusión - Bosques Aleatorios:")
print(confusion_matrix_forest)

#Considero que el mejor medelo para el conjunto de datos es ele de randomforest ya que tiene un menor número de errores sobre todo en los de falsos negativos

# Representación gráfica de la clasificación para ambos modelos
library(gridExtra)

# Gráfico de clasificación para el modelo de árboles de decisión
plot_tree <- ggplot(df.Social, aes(x = Age, y = EstimatedSalary, color = factor(predictions_tree))) +
  geom_point() +
  ggtitle("Clasificación - Árbol de Decisión")

# Gráfico de clasificación para el modelo de bosques aleatorios
plot_forest <- ggplot(df.Social, aes(x = Age, y = EstimatedSalary, color = factor(predictions_forest))) +
  geom_point() +
  ggtitle("Clasificación - Bosques Aleatorios")

# Construir una sola gráfica en grid que incluya los dos gráficos anteriores
grid.arrange(plot_tree, plot_forest, ncol = 2)
