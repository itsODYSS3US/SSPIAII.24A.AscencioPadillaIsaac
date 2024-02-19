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
