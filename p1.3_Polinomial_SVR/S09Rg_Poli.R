source("Clases/Lib.Preprocess.R")
getwd()
setwd("C:/Users/uliss/OneDrive/Documentos/UdG/SSP IA II/R")

set.seed(1991)
options(scipen = 999)

#Preprocesamiento
df.Position <- read.csv("Datasets/Position_Salaries.csv", header = T, stringsAsFactors = T)

df.Position$Position <- NULL
plot(df.Position)

#Split

#Modelos
#Modelo 1
mdl.lineal.Position <- lm(formula = Salary ~ Level, data = df.Position)

summary(mdl.lineal.Position)

plt.Position.g1 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.lineal.Position, newdata = df.Position)),
colour = "red")

#Modelo 2
mdl.poly2.Position <- lm(formula = Salary ~ poly(Level, 2), data = df.Position)
summary(mdl.poly2.Position)


plt.Position.g2 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.poly2.Position, newdata = df.Position)),
colour = "blue") 


#Modelo 3
mdl.poly3.Position <- lm(formula = Salary ~ poly(Level, 3), data = df.Position)
summary(mdl.poly3.Position)


plt.Position.g3 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.poly3.Position, newdata = df.Position)),
colour = "green") 

#Modelo 4
mdl.poly4.Position <- lm(formula = Salary ~ poly(Level, 4), data = df.Position)
summary(mdl.poly4.Position)


plt.Position.g4 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.poly4.Position, newdata = df.Position)),
colour = "violet")

#Modelo 5
mdl.poly5.Position <- lm(formula = Salary ~ poly(Level, 5), data = df.Position)
summary(mdl.poly5.Position)


plt.Position.g5 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.poly5.Position, newdata = df.Position)),
colour = "gray")

#Gráfico
plt.Position <- ggplot() +
geom_point(aes(x = df.Position$Level,y = df.Position$Salary))


plt.G1 <- plt.Position + plt.Position.g1 + ggtitle("Grado 1")
plt.G2 <- plt.Position + plt.Position.g2 + ggtitle("Grado 2")
plt.G3 <- plt.Position + plt.Position.g3 + ggtitle("Grado 3")
plt.G4 <- plt.Position + plt.Position.g4 + ggtitle("Grado 4")
plt.G5 <- plt.Position + plt.Position.g5 + ggtitle("Grado 5")


plt.G5


plt.grid <- plot_grid(plt.G1,plt.G2,plt.G3,plt.G4,plt.G5, cnol = 2)

plt.grid


ggsave(filename = "poli_lin.jpg", plot = plt.grid, units = "in", height = 7, width = 14)


#modelos extraños
mdl.raro.Position <- lm(formula = Salary ~ Level + I(Level^4), data = df.Position)
summary(mdl.raro.Position)


plt.Position.g5 <- geom_line(aes(x = df.Position$Level,
y = predict(mdl.raro.Position, newdata = df.Position)),
colour = "orange")




#SVR
install.packages("e1071")
library(e1071)

svr.lin <- svm(formula = Salary ~ ., data = df.Position, kernel = "line", type = "eps-regression")
summary(svr.lin)


#Gráfica
plt.svr <- ggplot() +
theme_light()+
geom_point(aes(x = df.Position$Level,y = df.Position$Salary))+
xlab("Nivel de puesto") +
ylab("Salario") +
geom_line(aes(x = df.Position$Level, y = predict(svr.lin, newdata = df.Position)), colour = "darkblue", linewidth = 1.5) +
geom_line(aes(x = df.Position$Level, y = predict(svr.poly, newdata = df.Position)), colour = "darkred", linewidth = 1.5) +
geom_line(aes(x = df.Position$Level, y = predict(svr.rad, newdata = df.Position)), colour = "violet", linewidth = 1.5) +
geom_line(aes(x = df.Position$Level, y = predict(svr.sig, newdata = df.Position)), colour = "orange", linewidth = 1.5)
plt.svr


svr.poly <- svm(formula = Salary ~ ., data = df.Position, kernel = "polynomial", type = "eps-regression")
summary(svr.poly)

svr.rad <- svm(formula = Salary ~ ., data = df.Position, kernel = "radial", type = "eps-regression")
summary(svr.rad)

svr.sig <- svm(formula = Salary ~ ., data = df.Position, kernel = "sigmoid", type = "eps-regression")
summary(svr.rad)



y_pred <- predict(svr.rad, newdata = df.Position)
y_pred



Y.pred <- predict(svr.rad, newdata = df.Position)
ssr <- sum((df.Position$Salary ~ Y.pred)^2)

Y.mean <- mean(df.Position$Salary)
sst <- sum((df.Position$Salary ~ Y.mean)^2)

r.squared <- 1 - (ssr / sst)

n <- length(df.Position$Salary)
p <- ncol(df.Position)

adf.r <- 1 - (1 - r.squared) * ((n-1)/(n-p-1))


summary(y_pred)
