# Isaac Ulises Ascencio Padilla
source("Lib.Preprocess.R")
getwd()
setwd("C:/Users/uliss/OneDrive/Documentos/UdG/SSP IA II/R/SSPIAII.24A.AscencioPadillaIsaac")

install.packages("rmarkdown")
library(rmarkdown)

set.seed(1991)
options(scipen = 999)


#Preprocesamiento
df.Startups <- read.csv("Datasets/50_Startups.csv", header = T, stringsAsFactors = T)

df.StartupsD <- one_hot(as.data.table(df.Startups))
df.StartupsD$'State_New York' <- NULL

df.StartupsNew <- as.data.frame(df.StartupsD)



#Selección de variables

mdl.StartupsAll <- lm(formula = Profit ~ ., data = df.StartupsNew)


library(MASS)

mdl.Startups <- stepAIC(mdl.StartupsAll, direction = "both", trace = 1)
summary(mdl.Startups)

#Separación de entrenamiento y prueba
Split <- sample.split(Y = df.StartupsNew$Profit, SplitRatio = 0.8)
df.Startups.Train <- subset(df.StartupsNew, Split == T)
df.Startups.Test <- subset(df.StartupsNew, Split == F)

#Modelos
#Modelo 1
mdl.poly.Startups <- lm(formula = Profit ~ poly(R.D.Spend, 2), data = df.Startups.Train)

summary(mdl.poly.Startups)

plt.Startups.g1 <- geom_line(aes(x = df.Startups$R.D.Spend,
y = predict(mdl.poly.Startups, newdata = df.Startups)),
colour = "red")

#Modelo 2
mdl.poly2.Startups <- lm(formula = Profit ~ poly(R.D.Spend, 3), data = df.Startups.Train)
summary(mdl.poly2.Startups)


plt.Startups.g2 <- geom_line(aes(x = df.Startups$R.D.Spend,
y = predict(mdl.poly2.Startups, newdata = df.Startups)),
colour = "blue") 


#Modelo 3
mdl.poly3.Startups <- lm(formula = Profit ~ poly(R.D.Spend, 4), data = df.Startups.Train)
summary(mdl.poly3.Startups)


plt.Startups.g3 <- geom_line(aes(x = df.Startups$R.D.Spend,
y = predict(mdl.poly3.Startups, newdata = df.Startups)),
colour = "green") 

#Modelo 4
mdl.poly4.Startups <- lm(formula = Profit ~poly(R.D.Spend, 5), data = df.Startups.Train)
summary(mdl.poly4.Startups)


plt.Startups.g4 <- geom_line(aes(x = df.Startups$R.D.Spend,
y = predict(mdl.poly4.Startups, newdata = df.Startups)),
colour = "violet") 


#Gráfico
plt.Startups <- ggplot() +
geom_point(aes(x = df.Startups.Train$R.D.Spend,y = df.Startups.Train$Profit))


plt.G1 <- plt.Startups + plt.Startups.g1 + ggtitle("Grado 2")
plt.G2 <- plt.Startups + plt.Startups.g2 + ggtitle("Grado 3")
plt.G3 <- plt.Startups + plt.Startups.g3 + ggtitle("Grado 4")
plt.G4 <- plt.Startups + plt.Startups.g4 + ggtitle("Grado 5")


plt.grid <- plot_grid(plt.G1,plt.G2,plt.G3,plt.G4, cnol = 2)

plt.grid

#Selección del modelo


smld <-  ggplot() +
  theme_light() +
  geom_point(aes(x = df.Startups$R.D.Spend, y = df.Startups$Profit)) +
  geom_line(aes(x = df.Startups$R.D.Spend, y = predict(mdl.poly2.Startups, newdata = df.Startups)), color = "darkblue")



#SVR
install.packages("e1071")
library(e1071)

svr.lin <- svm(formula = Profit ~ ., data = df.Startups.Train, kernel = "line", type = "eps-regression")
summary(svr.lin)


#Gráfica
plt.svr <- ggplot() +
theme_light()+
geom_point(aes(x = df.Startups.Train$R.D.Spend, y = df.Startups.Train$Profit))+
geom_line(aes(x = df.Startups.Train$R.D.Spend, y = predict(svr.lin, newdata = df.Startups.Train)), colour = "darkblue", linewidth = 1.5)
plt.svr




y_pred <- predict(svr.lin, newdata = df.Startups.Train)
y_pred

