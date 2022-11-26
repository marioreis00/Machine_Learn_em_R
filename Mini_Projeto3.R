#
# Mini Projeto 3
# Prevendo a Inadinplência de Clientes com Machine Learning e Power BI
#

#Definindo a pasta de trabalho
setwd("C:/Users/Mário/Documents/PowerBI/cap15/dados")
getwd()

#Instalação dos pacotes usados no projeto
#Se instala apenas 1 vez
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
#install.packages("randomForest") precisa usar versão anterior
install.packages("e1071")

#Carregar os pacotes
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(ramdomForest)
library(e1071)

#Carregar o dataset

dados_clientes <- read.csv("dados/dataset.csv")

#Visualizando os dados e sua estrutura
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

# Análise Explorativa, Limpeza e Trasformação dos dados

#Removendo a primeira coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#Renomeando a coluna Classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

#Converter os atributos genero | Escolaridade | Estado Civil | | Idade
#Mudar para valores (categorias)

colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

#Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero,
                         c(0,1,2),
                         labels = c("Masculino",
                                    "Feminino"))
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)


#Escolaridade
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
?cut
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                             c(0,1,2,3,4),
                             labels = c("Pos Graduado",
                                        "Graduado",
                                        "Ensino Medio",
                                        "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

#Estado civil
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
?cut
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)

#Idade para Faixa Etária
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
?cut
dados_clientes$Idade <- cut(dados_clientes$Idade,
                                   c(0,30,50,100),
                                   labels = c("Jovem",
                                              "Adulto",
                                              "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
View(dados_clientes)

#Converter a variavel que indica pagamentos para fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#Verificar Dataset após conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)
View(dados_clientes)

#Alterar a váriavel dependente para tipo valor
str(dados_clientes$inadimplente)
colnames(dados_clientes)
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)
View(dados_clientes)

# Total de Inadinplencia e não Inadinplentes
table(dados_clientes$inadimplente)

#Porcentagem entre classes
prop.table(table(dados_clientes$inadimplente))

#Plot de distribuição usando ggplot2
qplot(inadimplente, data = dados_clientes, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set Seend
set.seed(12345)
?set.seed

#Amostragem estratificada
#Seleciona as linhas de acordo com a variável inadinplemte como strata
?createDataPartition
indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list = FALSE)
dim(indice)

#Definidos os dados de treinamento como subconjunto do conjunto de dados original
#com numeros de indices de linha conforme acima e todas as colunas

dados_treino <- dados_clientes[indice,]
table(dados_treino$inadimplente)

#Porcentagem entre as Classes
prop.table(table(dados_treino$inadimplente))

#Registros no database de treino
dim(dados_treino)

#Comparando as porcentagens entre as classes de treinamento e dados originais
comparar_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                        prop.table(table(dados_clientes$inadimplente)))
colnames(comparar_dados) <- c("Treinamento", "Original")
comparar_dados

#Melt Data - Converte colunas em linhas
melt_comparar_dados <- melt(comparar_dados)
melt_comparar_dados

# Plot para ver treinamento x original
ggplot(melt_comparar_dados, aes(x = x1, y = value)) +
  geom_bar(aes(fill = x2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treino esta no de teste. Observe sinal de -(menos)
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)

############## 1º Modelo de Machine Learning #############

#Construindo 1ª vesão do modelo

View(dados_treino)
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

#Avaliando o modelo
plot(modelo_v1)

#Previsão com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

#Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1

#Calculando Precision, Recall e F1 Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Balanceamento de Classe
install.packages("DMwR")
library(DMwR)

#Aplicando o Smote - Smote: Synthetic Minority OVer-Sampling Technique

table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$inadinplente)
prop.table(table(dados_treino_bal$inadimplente))


# Construindo segunda versão do modelo
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2


#Avaliando o modelo
plot(modelo_v2)

#Previsão com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

#Confusion Matrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2

#Calculando Precision, Recall e F1 Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Importancia das váriaveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

#Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

#Criando o rank de variáveis baseado na importancia
rankImportance <- varImportance %>%
  mutate(Rank = paste0("#", dense_rank(desc(Importance))))

#Usando ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
                      y = Importance, 
                      fill = Importance)) +
  geom_bar(stat = 'identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
           hjust = 0,
           vjust = 0.55,
           size = 4,
           colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

# Construindo a terceira versão do modelo apenas com as variáveis mais importantes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3


#Avaliando o Modelo
plot(modelo_v3)

#Previsão com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

#Confusion Matrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3

#Calculando Precision, Recall e F1 Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


# Salvar o modelo em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

#Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")
modelo_final

#Prevendo inadimplencia com 3 clientes

#Add dados clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

#Concatenarem um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

#Previsão
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)

#Chegando os tipos dos dados
str(dados_treino_bal)
str(novos_clientes)

#Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)


#Previsão
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_client)
