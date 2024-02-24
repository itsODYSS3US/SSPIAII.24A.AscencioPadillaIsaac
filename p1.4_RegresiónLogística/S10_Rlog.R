# Isaac Ulises Ascencio Padilla

source("Lib.Preprocess.R")
getwd()
setwd("C:/Users/uliss/OneDrive/Documentos/UdG/SSP IA II/R/SSPIAII.24A.AscencioPadillaIsaac")

set.seed(1991)
options(scipen = 999)

#Datos
df.Social <- read.csv("Datasets/Social_Network_Ads.csv", header = T, stringsAsFactors = T)

df.Social$User.ID <- NULL
df.Social$Gender <- NULL

summary(df.Social)

#Exploración
boxplot(df.Social$EstimatedSalary)


#Factores
df.Social$Gender <- as.numeric(df.Social$Gender)

#Selección de variables
cor.Social <- cor(df.Social)


#Escalado
df.Social$Age <- scale(df.Social$Age)
df.Social$EstimatedSalary <- scale(df.Social$EstimatedSalary)


colnames(df.Social) <- c('Age','EstimatedSalary','Purchased')
df.Social$Age <- as.numeric(df.Social$Age)
df.Social$EstimatedSalary <- as.numeric(df.Social$EstimatedSalary)

Split <- sample.split(df.Social$Purchased, SplitRatio = 0.8)
df.Social.Train <- subset(df.Social, Split == T)
df.Social.Test <- subset(df.Social, Split == F)



#Modelo
mdl.Rlog <- glm(formula = Purchased ~ ., data = df.Social.Train, family = binomial)

summary(mdl.Rlog)


#Predicción
predict.Social <- predict(mdl.Rlog, type = "response", newdata = df.Social.Test)

Y.pred <- ifelse(predict.Social >= 0.5, 1, 0)
Y.pred


plt.log <- ggplot(df.Social.Train, aes(x = Age, y = Purchased)) +
  geom_point(alpha = 0.5) +
  stat_smooth(method = "glm", se = FALSE, method.args = list(family = binomial))


ggsave(filename = "log_lin.jpg", plot = plt.log, units = "in", height = 7, width = 14)


#Colorcitos
set <- df.Social.Test
x1 <- seq(min(set[,1]) - 1, max(set[,1]) + 1, by = 0.1) 
x2 <- seq(min(set[,2]) - 1, max(set[,2]) + 1, by = 0.1) 

grid.set <- expand.grid(x1, x2)

colnames(grid.set) <- c('Age', 'EstimatedSalary')

prod.set <- predict(mdl.Rlog, type = 'response', newdata = grid.set)

Y.grid <- ifelse(prod.set > 0.5, 1, 0)

plot(set[,-3], main = "Gráfico de clasificación", xlab = "Edad", ylab = "Suedo estimado", xlim = range(x1), ylim = range(x2))

contour(x1, x2, matrix(as.numeric(Y.grid), length(x1), length(x2)), add = T)

points(grid.set, pch =".", col = ifelse(Y.grid == 1, "darkblue", "darkred"))
points(set, pch = 21, bg = ifelse(set[,3] == 1, "darkblue", "darkred"))


#Evaluación del modelo
#Check imbalanced

summary(as.factor(df.Social.Train$Purchased))

ggplot(df.Social.Train, aes(x=as.factor(df.Social.Train$Purchased))) + geom_bar(stat = "count") + theme_minimal()


#Matriz de confusión
matriz <- table(df.Social.Test$Purchased, Y.pred)
matriz

confusionMatrix(as.factor(Y.pred), as.factor(df.Social.Test$Purchased), mode = "everything", positive = "1")

confusionMatrix(matriz, positive = "1")$byClass

TP <- matriz[2,2]
FP <- matriz[1,1]
TN <- matriz[2,1]
FN <- matriz[1,2]


accurance <- (TP + TN) / (TP + TN + FP + FN)
accurance

precision <- TP / (TP + FP)
precision

recall <- TP / (TP + FN)
recall

f1Score <- (2 * precision * recall) / (precision + recall)  
f1Score

#Formulas
#accurance = 0.3625
#precision = 0.3076923
#recall = 0.7692308
#f1Score = 0.4395604

#Librerias
#accurance = 0.7860
#precision = 0.7692
#recall = 0.6897
#f1Score = 0.7273

#Conclusión
# Los valores obtenidos de forma manual y con el uso de la librería varían mucho y pienso que esto se puede deber a que los valores pueden variar
# según la fuente y el contexto, investigando encontré otra fórmula diferente para calcular el f1 score por lo que puede que las librerías 
# utilicen algo diferente para calcular los valores.