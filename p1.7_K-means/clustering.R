# Isaac Ulises Ascencio Padilla

source("Lib.Preprocess.R")
getwd()
setwd("C:/Users/uliss/OneDrive/Documentos/UdG/SSP IA II/R/SSPIAII.24A.AscencioPadillaIsaac")

set.seed(1991)
options(scipen = 999)

df.Credit <- read.csv("Datasets/Credit Card_Kaggle.csv", header = T, stringsAsFactors = T)
df.Credit$CUST_ID <- NULL

df.Credit.new <- na.omit(df.Credit)

cor.Credit <- cor(df.Credit.new)
cor.Credit

df.Card <- NULL
df.Card$TENURE <- df.Credit.new$TENURE
df.Card$PURCHASES <- df.Credit.new$PURCHASES
df.Card$ONEOFF_PURCHASES <- df.Credit.new$ONEOFF_PURCHASES
df.Card$PURCHASES_INSTALLMENTS_FREQUENCY <- df.Credit.new$PURCHASES_INSTALLMENTS_FREQUENCY
df.Card$PURCHASES_FREQUENCY <- df.Credit.new$PURCHASES_FREQUENCY
df.Card$CASH_ADVANCE_FREQUENCY <- df.Credit.new$CASH_ADVANCE_FREQUENCY
df.Card$CASH_ADVANCE_TRX <- df.Credit.new$CASH_ADVANCE_TRX

boxplot(df.Card$PURCHASES)

df.Card <- as.data.frame(df.Card)

#Elbow
n.obs <- length(df.Card)

wcss <- vector()
for(i in 1:15){
    wcss[i] <- kmeans(df.Card, i)$tot.withinss
}

wcss <- as.data.frame(wcss)

wcss$k <- seq(1,15,1)

#Gráfica del codo
ggplot()+
geom_line(aes(x = wcss$k, y = wcss$wcss))+
geom_point(aes(x=wcss$k, y=wcss$wcss, color=wcss$k))+ggtitle("Método del codo")+
xlab("Iteración")+
ylab("wcss")


library(cluster)
install.packages("factoextra")
library(factoextra)

install.packages("plotlly")
library(plotly)

#Definimos el número de k, que representa el número de centroides que tendrá el gráfico
# Ejecuta K-means en el dataframe df.Card con 4 clusters
kmeans_clusters <- kmeans(df.Card, centers = 4)

# Visualización con plot_ly
# Utiliza plot_ly para crear una visualización interactiva
plot_ly(data = df.Card,               # Los datos para visualizar
        x = ~PURCHASES,               # Variable para el eje X (Compras totales)
        y = ~ONEOFF_PURCHASES,       # Variable para el eje Y (Compras únicas)
        color = factor(kmeans_clusters$cluster)) %>%  # Variable para el color, asignada según los clusters obtenidos
  add_markers() %>%                   # Añade puntos al gráfico
  layout(title = "Gráfico de clusters",          # Título del gráfico
         xaxis = list(title = "Compras totales"),    # Etiqueta del eje X
         yaxis = list(title = "Compras únicas"))     # Etiqueta del eje Y


# Utiliza fviz_cluster para visualizar los resultados con elipses y puntos
fviz_cluster(kmeans_clusters, data = df.Card, geom = "point", ellipse.type = "t",
             pointsize = 2, palette = "jco", ggtheme = theme_minimal())


install.packages("ggfortify")
library(ggfortify)
# Visualización de clusters utilizando ggfortify
# Utiliza autoplot para visualizar los resultados
autoplot(kmeans_clusters, data = df.Card, frame = TRUE, frame.type = "norm")
