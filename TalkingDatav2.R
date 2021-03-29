# Projeto 1 - Detecção de fraudes em cliques em propaganda - TalkingData
# Projeto desenvolvido no curso Formação Cientista de Dados da DataScience Academy
# https://www.datascienceacademy.com.br/bundles?bundle_id=formacao-cientista-de-dados
# Dados disponíveis em: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
# Carlos Eduardo Carvalho
# LinkedIn: https://www.linkedin.com/in/carlos-carvalho-93204b13/

# Definindo o diretório de trabalho
setwd("D:/CIENTISTA_DADOS/1_BIG_DATA_R_AZURE/PROJETO1")
getwd()

install.packages("kernlab")
library(ROSE)
library(caret) # Confusion matrix
library(readr)
library(data.table) # fread
library(tidyverse) # glimpse
library(caTools)
library(plyr) # count
library(corrplot)
library(corrgram)
library(psych)
library(randomForest)
library(e1071)
library(kernlab) # Supor Vector Machine


# Como os datasets train e test são muito grandes, resolvi utilizar apenas o dataset train_sample
# que possui 100 mil linhas.
df <- fread("train_sample.csv")
View(df)

glimpse(df)

# Vou retirar a variável attributed_time, pois ela parece redundante em relação a variável alvo.
df <- df[,-7]

# Pode se ver que a variável target está muito desequilibrada.
table(df$is_attributed)

# Fazendo um histograma para observar o desequilíbrio entre os valores da variável target
hist(df$is_attributed, breaks = 10, main = "Dados Desbalanceados",col = c("tomato","blue"))

# Agora é necessário fazer um balanceamento no dataset para que o algoritmo não fique tendencioso.
df2 <- ovun.sample(is_attributed~., data=df,
                   p=0.5, 
                   method="both")$data
table(df2$is_attributed)
hist(df2$is_attributed, breaks = 10, main = "Dados Balanceados",col = c("tomato","blue"))

amostra <- sample.split(df2$is_attributed, SplitRatio = 0.1)

df2 = subset(df2, amostra == TRUE)
View(df2)


amostra <- sample.split(df2$is_attributed, SplitRatio = 0.70)
# Criando dados de treino - 70% dos dados
treino = subset(df2, amostra == TRUE)

# Criando dados de teste - 30% dos dados
teste = subset(df2, amostra == FALSE)
View(teste)


##################################################################################################
# O primeiro modelo testado será o de regressão logística, com variáveis numéricas
modelo_RL_1 <- glm(is_attributed ~ ., data = treino, family = "binomial")

# Visualizando o modelo
summary(modelo_RL_1)
# Testando o modelo nos dados de teste
prevendo_RL_1 <- predict(modelo_RL_1, teste, type="response")
prevendo_RL_1 <- round(prevendo_RL_1)


test.class.var <- teste[,7]
CF_1 <- confusionMatrix(table(data = prevendo_RL_1, reference = test.class.var), positive = '1')
CF_1$table
CF_1$overall["Accuracy"]

# Visualizando os valores previstos e observados
resultados_RL_1 <- cbind(prevendo_RL_1, teste$is_attributed) 
colnames(resultados_RL_1) <- c('Previsto','Real')
resultados_RL_1 <- as.data.frame(resultados_RL_1)
View(resultados_RL_1)

# Tentei fazer o modelo transformando  as variáveis para fator, mas o R ficou rodando e não deu
# nenhum resultado.


#####################################################################################################
# Segundo modelo será Naive Bayes 
modelo_NB_1 <- naiveBayes(is_attributed ~ .,treino)
print(modelo_NB_1)
prevendo_NB_1 <- predict(modelo_NB_1, teste)
head(prevendo_NB_1)
table(prevendo_NB_1, true = teste$is_attributed)

test.class.var <- teste[,7]
CF_2 <- confusionMatrix(table(data = prevendo_NB_1, reference = test.class.var), positive = '1')
CF_2$table
CF_2$overall["Accuracy"]


########################################################################################################
##### Modelo Suport Vector Machine com todas as variáveis- SVM
modelo_SVM_1 <- ksvm(is_attributed ~ .,data = treino, kernel="vanilladot" )
prevendo_SVM_1 <- predict(modelo_SVM_1, teste)
prevendo_SVM_1 <- round(prevendo_SVM_1)
View(prevendo_SVM_1)
table(prevendo_SVM_1, true = teste$is_attributed)

# Aparentemente o modelo previu alguns valores como sendo 2, que é uma condição que não existe.
# Então vou transformar esses resultados 2 em 1.
for (i in 1:length(prevendo_SVM_1)){
  if(prevendo_SVM_1[i] == 2){
    prevendo_SVM_1[i] = 1
  }
}
table(prevendo_SVM_1, true = teste$is_attributed)

test.class.var <- teste[,7]
CF_3 <- confusionMatrix(table(data = prevendo_SVM_1, reference = test.class.var), positive = '1')
CF_3$table
CF_3$overall["Accuracy"]

# Data frame com a acurácia dos modelos testados
accuracyVector <- c(CF_1$overall["Accuracy"],CF_2$overall["Accuracy"], CF_3$overall["Accuracy"])

# Criando um dataframe com todas as acurácias conseguidas nos 6 modelos testados.
Modelos <- c("Regressao Logistica", "Naive Bayes", "SVM")
accuracyDataFrame <- data.frame(Modelos, accuracyVector)
colnames(accuracyDataFrame) <- c("Modelos", "Acurácia")
View(accuracyDataFrame)

# O modelo que melhor respondeu aos dados foi o Naive Bayes.
# Agora vou tentar algumas modificações para ver se é possível aumentar a acurácia.
# Primeiro vou tentar usar as variáveis do tipo fator
## Criando funções para converter variáveis categóricas para tipo fator.
glimpse(df2)

to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}
v <- c("ip", "app", "device", "os", "channel", "is_attributed")
df3 <- to.factors(df2, v)
glimpse(df3)

# Vou fazer a divisão em treino e teste com as variáveis do tipo fator
amostra <- sample.split(df3$is_attributed, SplitRatio = 0.70)
# Criando dados de treino - 70% dos dados
treinof = subset(df3, amostra == TRUE)

# Criando dados de teste - 30% dos dados
testef = subset(df3, amostra == FALSE)
# Vou aplicar novamente o modelo Naive Bayes

modelo_NB_2 <- naiveBayes(is_attributed ~ .,treinof)
print(modelo_NB_2)
prevendo_NB_2 <- predict(modelo_NB_2, testef)
head(prevendo_NB_2)
table(prevendo_NB_2, true = testef$is_attributed)

test.class.var <- testef[,7]
CF_4 <- confusionMatrix(table(data = prevendo_NB_2, reference = test.class.var), positive = '1')
CF_4$table
CF_4$overall["Accuracy"]

# É possível verificar que o modelo Naive Bayes apresentou uma boa resposta com as variáveis do tipo fator
# alcançando uma acurácia de 96%.
# Portanto, esse modelo será o escolhido para detecção de fraudes no tráfico de cliques em propagandas 
# de aplicações mobile
df_final <- as.data.frame(prevendo_NB_2)
View(df_final)
click_id <- seq(from = 1, to = length(df_final$prevendo_NB_2))
df_final <- cbind(click_id, df_final)
colnames(df_final) <- c("click_id", "is_attributed")


# Salva em um arquivo csv
write.csv(df_final, "df_final.csv")

