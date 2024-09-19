# options(digits=2)
rm(list=ls()) #borro todas las variables del workspace (rm)
# setwd('/home/juank/Repos/LSA/R/')
setwd('/home/brunobian/Documents/Repos/LSA/R')

library(lme4)
library(plyr)
library(dplyr)
library(ggplot2)
library(lattice)
library(RColorBrewer)
library(reshape2)
library(corrplot)
library(gridExtra)


source('functions/loadDataSets.R')
source('functions/jk.logit.R')
source('functions/bb.logit.R')
source('functions/remef.v0.6.10.R')
figFolder = '/home/brunobian/Dropbox/Labo/reading/9 manuscritos/Predicting Predictability/Figures/Round 1/1-originales/'

##### Load DataFrames #####################################################################################################
# Load Eye data
# Esto viene de la ultima version del paper de Repeticiones. 
# Copie los archivos necesarios para generarlo aca.
load("eye_data/data_without_skips_2019.Rda")
# summary(data)

# Load Cloze data
# Esto viene de Gaston, solo uso la data Cloze
# load("pred_data/dataCLOZE.Rda")
# summary(dataCLOZE)
# data$CLOZE_pred <- jk.logit(dataCLOZE$CLOZE_pred)

cloze <- read.csv("../cloze/predictability2.csv")
cloze <- bb.logit(cloze)
data  <- addCloze(data, cloze)
data  <- data[data$CLOZE_nPred >= 8,]
data  <- data[data$CLOZE_nPred_next >= 8,]

data   <- loadEyeData(data)

# Merge DataFrames 
# Load LSA data and Add LSA to the Eye data
# - Las últimas 15 columnas son de LSA: 
# 'Texto': nombre del texto (basta de números)
# 'Palabra Target': palabra 
# 'wng (inicia en 1)': posicion absoluta en el texto 
# 'Tamaño ventana': ventana, coincide con el numero del archivo
# 'Pals Ventana Con SW': palabras en la ventana que incluye stopwords 
# 'Pals Ventana Sin SW': palabras en la ventana que no incluye stopwords 
# 'N Pals Contenido': cantidad de palabras de contenido en la ventana que incluye stopwords (si la ventana es de 50, son 50 palabras totales, N palabras de contenido)
# 'D resultante conSW': distancia de la palabra target a la resultante de la ventana con SW
# 'D resultante sinSW': distancia de la palabra target a la resultante de la ventana sin SW
# 'D promedio conSW': promedio de las  entre la palabra target y cada una de las palabras de la ventana con SW
# 'D promedio sinSW': promedio de las  entre la palabra target y cada una de las palabras de la ventana sin SW 
# 'D minima conSW': distancia mínima dentro de la ventana
# 'D minima sinSW': distancia mínima dentro de la ventana
# 'd_w2v_sinSW': distancia de w2v en ventana sin SW
# 'd_w2v_conSW': distancia de w2v en ventana sin SW
# 'p_cercanas_sinSW': proporcion de palabras cercanas en LSA en ventana sin SW
# 'p_cercanas_conSW': proporcion de palabras cercanas en LSA  en ventana con SW

# A cada elemento de data hay que asignarle los nuevos regresores que me interesen.
ventanas = c('002','003','004','005',
             '006','007','008','009',
             '010','011','012',
             '015','017','020','030','040','050',
             '060','100','150')

# ventanas = c('005','015','025','050',
#              '100','150')

for (v in ventanas) {
  file <- paste0("../distancias/usandoSW_FastText_corpus/ventana",v, ".csv")
  LSA  <- read.csv(file, comment.char = "") # Tiene todas, aun las palabras que no se repitieron
  LSA$d_w2v_sinSW  <- as.numeric(LSA$d_w2v_sinSW)
  data <- addLsaW2vToData(data, LSA, v)
}

ngrams4 <- read.csv('~/Documents/Repos/ngrams/TABLE_complete2')
data <- addNgramsToData(data, ngrams4)
data <- generateNgramCache(data)

# Elimino pals con NA
dataAll <- data 

# load(file="FullData.rda")
dataAll -> data

data[data==-10] <- NA
tmp <- c()
for (i in colnames(data)){
  tmp <- c(tmp,sum(is.na(data[,i])))
}
hist(tmp,100)
# Elimino las columnas que me generan muchos NA
colnames(data)[tmp>20000]
data <- data[,colnames(data)[tmp<20000]]

# Elimino todas las filas con al menos un NA, 
# para que todos los modelos corran con los mismos datos
data <- data[complete.cases(data),]

save(file="FullData.rda",list="dataAll")
save(file="DataFilt.rda",list="data")
##### Load filtered DataFrame  #####################################################################################################
rm(list=ls()) #borro todas las variables del workspace (rm)
load(file="DataFilt.rda")

#####[Fig 1: Pred descriptivo]#####################################################################################################
# Correlación nRep - Pred
dataRep  <- data[(data$nREP < 9),]
tableRep <- ddply(dataRep, .(nREP), summarise, 
                    N=length(CLOZE_pred), 
                    M=mean(CLOZE_pred), 
                    SD=sd(CLOZE_pred), 
                    SE=SD/sqrt(N) )

ggplot(tableRep, aes(x = nREP, y = M, colour = 1)) +
  geom_errorbar(aes(ymin = M-SE, ymax = M+SE)) +
  geom_point() + geom_line()+
  theme(legend.position = "none")+
  xlab('Repetition number') + 
  ylab('Logit cloze-Predictability')
ggsave(file=paste0(figFolder, "2a-Pred-Rep.eps"))

# Correlación log(freq) - Pred
dataFreq  <- data
dataFreq$freq_round <- round(data$freq_noncentered*4)/4
tableFreq <- ddply(dataFreq, .(freq_round), summarise,
                    N=length(CLOZE_pred), 
                    M=mean(CLOZE_pred), 
                    SD=sd(CLOZE_pred), 
                    SE=SD/sqrt(N) )

ggplot(tableFreq, aes(x = freq_round, y = M, colour = 1)) +
  geom_errorbar(aes(ymin = M-SE, ymax = M+SE)) +
  geom_point() + geom_line()+
  theme(legend.position = "none")+
  xlab('Log10 Frequency') + 
  ylab('Logit cloze-Predictability')
ggsave(file=paste0(figFolder, "2b-Pred-Freq.eps"))

# Correlación log(length) - Pred
dataLength  <- data[(data$invlength_noncentered > 0.07),]
tableLength <- ddply(dataLength, .(invlength_noncentered), summarise, 
                       N=length(CLOZE_pred), 
                       M=mean(CLOZE_pred), 
                       SD=sd(CLOZE_pred), 
                       SE=SD/sqrt(N) )

ggplot(tableLength, aes(x = invlength_noncentered, y = M, colour = 1)) +
  geom_errorbar(aes(ymin = M-SE, ymax = M+SE)) +
  geom_point() +
  geom_line()+
  theme(legend.position = "none")+
  xlab('Inverse Length') + 
  ylab('Logit cloze-Predictability')
ggsave(file=paste0(figFolder, "2c-Pred-length.eps"))

# Correlación rps - Pred
dataRps <- data
dataRps$rps_round <- round(dataRps$rps_noncentered*5)/5
tableRps <- ddply(dataRps, .(rps_round), summarise, 
                     N=length(CLOZE_pred), 
                     M=mean(CLOZE_pred), 
                     SD=sd(CLOZE_pred), 
                     SE=SD/sqrt(N) )

ggplot(tableRps, aes(x = rps_round, y = M, colour = 1)) +
  geom_errorbar(aes(ymin = M-SE, ymax = M+SE)) +
  geom_point() +
  geom_line()+
  theme(legend.position = "none")+
  xlab('Relative Position in Sentence') + 
  ylab('Logit cloze-Predictability')
ggsave(file=paste0(figFolder, "2d-Pred-rps.eps"))


#####[Fig S1: optimizo ngram]###################################################################################################
# Optimizo N
runOpt = FALSE # tarda mucho el optimizardor de delta y lambda. If FALSE, carga ya corrido

enes=1:7
correlation = c()
for (n in enes){
  name=paste0('X',n,'.gram')
  correlation = c(correlation,cor(ngrams4$cloze_predictor, ngrams4[name]))
}
df = data.frame(enes, correlation)

ggplot(df, aes(x = enes, y = correlation, colour = 1)) +
  geom_point() + geom_line()+
  theme(legend.position = "none")+
  xlab('N') + 
  ylab('Correlation with predictability')
ggsave(file=paste0(figFolder, "S1a-N-correlation.eps"))

# optimizo Delta y lambda
if (runOpt){
  columnas  <- colnames(data) 
  fields    <- columnas[grepl('X4.gramcache.',columnas) & !grepl('_next',columnas)]
  
  aicngram <- tvalngram <- deltas <- lambdas  <- c()
  for (f in fields){
    
    # Extraigo el delta y el lambda del nombre
    todoJunto <- strsplit(f,'X4.gramcache.')[[1]][2]
    separado  <- strsplit(todoJunto,'_')[[1]]
    deltas    = c(deltas,  as.numeric(separado[1]))
    lambdas   = c(lambdas, as.numeric(separado[2]))  
    
    # Corro este modelo
    str = 'ngram'
    model = modelo(str, f,data,FALSE)
    aicngram  = c(aicngram, model[1])
    tvalngram = c(tvalngram, model[2])
  }
  
  df <- data.frame(aicngram, tvalngram, deltas,lambdas)
  save(df, file='Optim_delta_lambda2.rda')
} else {
  load(file='Optim_delta_lambda2.rda')
}

AICm <- dcast(data = df,formula = lambdas~deltas,value.var = "aicngram")
rownames(AICm) <- AICm[,1]
AICm[,1] <- NULL
AICm <- data.matrix(AICm)
levelplot(AICm[,2:dim(AICm)[2]],
          xlab = 'lambda',
          ylab = 'delta',
          main = 'Model AIC',
          col.regions = colorRampPalette(brewer.pal(9,"Reds"))(16))
inds = which(AICm == min(AICm), arr.ind = T)
print(paste('delta:', colnames(AICm)[inds[2]], 'lambda:', rownames(AICm)[inds[1]]))

pdf(paste0(figFolder, "S1b-delta-lambda-optimization.pdf"))
print(a)
dev.off()

tvalm <- dcast(data = df,formula = lambdas~deltas,value.var = "tvalngram")
rownames(tvalm) <- tvalm[,1]
tvalm[,1] <- NULL
tvalm <- data.matrix(tvalm)
a <- levelplot(tvalm[,2:dim(tvalm)[2]],
          xlab = 'lambda',
          ylab = 'delta',
          main = 'Model tval',
          col.regions = colorRampPalette(brewer.pal(9,"Reds"))(16))
inds = which(AICm == min(AICm), arr.ind = T)
print(paste('delta:', colnames(AICm)[inds[2]], 'lambda:', rownames(AICm)[inds[1]]))



#####[Fig S2: analizo Embeddings]####################################################################################################

# Corro el modelo base para comparar
M <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
            rpl + rpt + rps + 
            (1|sujid) + (1|textid) + (1|wordid), data = data)
aicBL  <- extractAIC(M)[2]
tvalBL <- coef(summary(M))[,"t value"]['este']

# Inicializo vars
aicResult  <- tvalResult <- aicProm <- tvalProm <- c()
aicMin <- tvalMin <- aicw2v <- tvalw2v <- c()
corResult <- corProm <- corMin <- corw2v <- c()
aicpCercanas  <- tvalpCercanas <- corpCercanas <- c()
aicFT_w <- tvalFT_w  <- corFT_w <- c()
aicFT_c <- tvalFT_c  <- corFT_c <- c()

source('functions/loadDataSets.R')
# Corro modelos para todas las variables
n1 <- FALSE
for (v in ventanas) {
  
  # LSA pCercanas
  str = 'pCercanas'
  model = modelo(str, v, data, n1)
  aicpCercanas  = c(aicpCercanas,  model[1])
  tvalpCercanas = c(tvalpCercanas, model[2])
  corpCercanas  = c(corpCercanas,  model[3])
  
  # LSA Resultante
  str = 'resultante'
  model = modelo(str, v, data, n1)
  aicResult  = c(aicResult,  model[1])
  tvalResult = c(tvalResult, model[2])
  corResult  = c(corResult,  model[3])
  
  # LSA promedio
  str = "promedio"
  model = modelo(str, v, data, n1)
  aicProm  = c(aicProm, model[1])
  tvalProm = c(tvalProm, model[2])
  corProm  = c(corProm,  model[3])
  
  # LSA minima
  str = "minima"
  model = modelo(str, v, data, n1)
  aicMin  = c(aicMin, model[1])
  tvalMin = c(tvalMin, model[2])
  corMin  = c(corMin,  model[3])
  
  # w2v promedio
  str = 'w2v'
  model = modelo(str, v, data, n1)
  aicw2v  = c(aicw2v, model[1])
  tvalw2v = c(tvalw2v, model[2])
  corw2v  = c(corw2v,  model[3])
  
  # FT wikipedia
  str = 'FT_w'
  model = modelo(str, v, data, n1)
  aicFT_w  = c(aicFT_w, model[1])
  tvalFT_w = c(tvalFT_w, model[2])
  corFT_w  = c(corFT_w,  model[3])
  
  # FT corpus
  str = 'FT_c'
  model = modelo(str, v, data, n1)
  aicFT_c  = c(aicFT_c, model[1])
  tvalFT_c = c(tvalFT_c, model[2])
  corFT_c  = c(corFT_c,  model[3])
  
}

n = as.numeric(ventanas)#c(3,4,5,10,20,30,50,75,100)
df = data.frame(n, aicResult, tvalResult, corResult,
                aicProm,   tvalProm,   corProm,
                aicMin,    tvalMin,    corMin,
                aicw2v,    tvalw2v,    corw2v,
                aicFT_w,   tvalFT_w,   corFT_w,
                aicFT_c,   tvalFT_c,   corFT_c,
                aicpCercanas, tvalpCercanas, corpCercanas)


cutoff <- data.frame( x = c(-Inf, Inf), y = 0.00 )
ggplot(df, aes(n)) + 
  geom_line(aes(y = corResult, color = 'LSA (Resultant)')) + 
  geom_line(aes(y = corProm, color = 'LSA (mean)')) + 
  geom_line(aes(y = corw2v, color = 'word2vec'))+ 
  geom_line(aes(y = corFT_w, color = 'FastText wiki'))+ 
  geom_line(aes(y = corFT_c, color = 'FastText corpus'))+ 
  geom_line(aes(y = corpCercanas, color = 'pCercanas'))+
  geom_line(aes(x,y, color = 'Delete!!'),cutoff)
ggsave(file=paste0(figFolder, "S2a-embeddings-Correlation.eps"))  

cutoff <- data.frame( x = c(-Inf, Inf), y = aicBL )
ggplot(df, aes(n)) + 
  geom_line(aes(x=n, y = aicResult, color = 'LSA (Resultant)')) +
  geom_line(aes(x=n, y = aicProm, color = 'LSA (mean)')) +
  # geom_line(aes(y = aicMin, color = 'LSA (min)')) +
  geom_line(aes(y = aicw2v, color = 'word2vec')) +
  geom_line(aes(y = aicFT_w, color = 'FastText wiki')) +
  geom_line(aes(y = aicFT_c, color = 'FastText corpus')) +
  geom_line(aes(y = aicpCercanas, color = 'pCercanas'))+ 
  geom_line(aes(x,y, color = 'Baseline'),cutoff)
ggsave(file=paste0(figFolder, "S2b-embeddings-AIC.eps"))

cutoff1 <- data.frame( x = c(-Inf, Inf), y =  1.96 )
cutoff2 <- data.frame( x = c(-Inf, Inf), y = -1.96 )
ggplot(df, aes(n)) + 
  geom_line(aes(y = tvalResult, color = 'LSA (Resultant)')) + 
  geom_line(aes(y = tvalProm, color = 'LSA (mean)')) + 
  # geom_line(aes(y = tvalMin, color = 'LSA (min)')) + 
  geom_line(aes(y = tvalw2v, color = 'word2vec')) +
  geom_line(aes(y = tvalFT_w, color = 'FastText wiki')) +
  geom_line(aes(y = tvalFT_c, color = 'FastText corpus')) +
  geom_line(aes(y = tvalpCercanas, color = 'pCercanas'))+ 
  geom_line(aes(x,y, color = 'Significance'), cutoff1) + 
  geom_line(aes(x,y, color = 'Significance'), cutoff2)
ggsave(file=paste0(figFolder, "S2c-embeddings-Tvals.eps"))


#####[Table 1: modelos en N]#####################################################################################################
# M0 Baseline model
M0_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M1 Baseline model + Cloze
M1_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps +
             CLOZE_pred +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M2 Baseline model + ngramCache
M2_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             X4.gramcache.0.0001500000_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M3 Baseline model + LSA009
M3_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             LSA009.promedio.conSW +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M4 Baseline model + ngramCahche + LSA 
M4_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M4 Baseline model + ngramCahche + LSA 
M5_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             LSA009.promedio.conSW +
             X4.gramcache.0.0001500000_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M6 Baseline model + ngramCahche + FT50
M6_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki +
             X4.gramcache.0.0001500000_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M7 Baseline model + LSA + FT50
M7_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki +
             LSA009.promedio.conSW +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M8 Baseline model + ngramCahche + FT50 + LSA15
M8_N <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki +
             LSA009.promedio.conSW +
             X4.gramcache.0.0001500000_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

anovas <- data.frame("M0.N" = c(NA,NA),
                      "M1.N" = c(anova(M0_N,M1_N)["M1_N","Pr(>Chisq)"], NA),
                      "M2.N" = c(anova(M0_N,M2_N)["M2_N","Pr(>Chisq)"], anova(M1_N,M2_N)["M2_N","Pr(>Chisq)"]),
                      "M3.N" = c(anova(M0_N,M3_N)["M3_N","Pr(>Chisq)"], anova(M1_N,M3_N)["M3_N","Pr(>Chisq)"]),
                      "M4.N" = c(anova(M0_N,M4_N)["M4_N","Pr(>Chisq)"], anova(M1_N,M4_N)["M4_N","Pr(>Chisq)"]),
                      "M5.N" = c(anova(M0_N,M5_N)["M5_N","Pr(>Chisq)"], anova(M1_N,M5_N)["M5_N","Pr(>Chisq)"]),
                      "M6.N" = c(anova(M0_N,M6_N)["M6_N","Pr(>Chisq)"], anova(M1_N,M6_N)["M6_N","Pr(>Chisq)"]),
                      "M7.N" = c(anova(M0_N,M7_N)["M7_N","Pr(>Chisq)"], anova(M1_N,M7_N)["M7_N","Pr(>Chisq)"]),
                      "M8.N" = c(anova(M0_N,M8_N)["M8_N","Pr(>Chisq)"], anova(M1_N,M8_N)["M8_N","Pr(>Chisq)"]))
row.names(anovas) <- c("M0.N","M1.N")
corrplot(as.matrix(anovas), is.corr = FALSE, method = "color", addgrid.col = TRUE,
         addCoef.col = TRUE,
         na.label = '-',
         col=colorRampPalette(c("blue","white","red"))(200),
         tl.col = 'black',tl.srt= 0)    

CI_M0_N <- confint(M0_N)
CI_M1_N <- confint(M1_N)
CI_M2_N <- confint(M2_N)
CI_M3_N <- confint(M3_N)
CI_M4_N <- confint(M4_N)
CI_M5_N <- confint(M5_N)
CI_M6_N <- confint(M6_N)
CI_M7_N <- confint(M7_N)
CI_M8_N <- confint(M8_N)

df <- data.frame(effect=c(''))
df_IC_N <- data.frame(effect=c(''))
for (i in c(0:8)){
  
  # t_values 
  modelName <- paste0('M',i,'_N')
  
  tmp <- eval(parse(text=modelName))
  ts <- coef(summary(tmp))[,"t value"]
  ts <- data.frame(effect=names(ts), ts)
  colnames(ts)[ncol(ts)] <- modelName
  
  df <- merge(df, ts, by = 'effect', all = TRUE)
  
  # estimates
  es <- coef(summary(tmp))[,"Estimate"]
  es <- data.frame(effect=names(es), es)
  colnames(es)[ncol(es)] <- paste0(modelName, "_b")
  
  # intervalos de confianza
  CIName <- paste0('CI_M',i,'_N')
  tmp <- eval(parse(text=CIName))
  ci <- data.frame(effect=row.names(tmp), tmp)
  colnames(ci) <- c("effect", paste0(modelName, "_2.5"), paste0(modelName, "_97.5"))
  ci <- ci[5:dim(ci)[1],]
  
  ci <- merge(es,ci, by = 'effect', all = TRUE)
  df_IC_N <- merge(df_IC_N, ci, by = 'effect', all = TRUE)
}
rownames(df) <- df[,1]
df[,1] <- NULL 
rownames(df_IC_N) <- df_IC_N[,1]
df_IC_N[,1] <- NULL 
df_IC_N <- df_IC_N[-1,]# La primera fila esta vacia por construccion
format(df_IC_N, digits=1)
write.csv(df_IC_N, paste0(figFolder,'S3-IC_N.csv'))

# Corro los modelos remef y los voy agregando al df
for (i in c(2:8)){
  modelName <- paste0('M',i,'_N')
  tmp <- eval(parse(text=modelName))
  
  fixef = rownames(coef(summary(tmp)))
  data$remefData <- remef(tmp, fix = fixef)
  Mr <- lmer(remefData ~ CLOZE_pred
              + (1|sujid) + (1|textid) + (1|wordid), 
              data = data, REML = FALSE)

  lst <- coef(summary(Mr))[,"t value"]
  df['BORRAR',modelName] <- NA
  df['CLOZE_pred_remef',modelName] <- lst['CLOZE_pred']
}
df <- df[-1,]# La primera fila esta vacia por construccion
df <- df[!rownames(df) %in% '(Intercept)',] # No me interesa mostrar el intercept (me saca de rango los colores)

df <- data.matrix(df)

pdf(paste0(figFolder, "3-TablaModelos.pdf"), width=13,height = 13)
corrplot(df, is.corr = FALSE, method = "color", addgrid.col = TRUE,
         addCoef.col = TRUE,
         na.label = '-',
         col=colorRampPalette(c("blue","white","red"))(200),
         tl.col = 'black',tl.srt= 0)
dev.off()

# Grafico AICs de los modelos 1-7
AICm0 = extractAIC(M0_N)[2]
aics = c(extractAIC(M1_N)[2],
         extractAIC(M2_N)[2],
         extractAIC(M3_N)[2],
         extractAIC(M4_N)[2],
         extractAIC(M5_N)[2],
         extractAIC(M6_N)[2],
         extractAIC(M7_N)[2],
         extractAIC(M8_N)[2]) - AICm0

pdf(paste0(figFolder, "3-AICModelos.pdf"))
barplot(aics, ylim = c(min(aics)-100, max(aics)+100),xpd = FALSE, 
        ylab = c('AIC(Mi) - AIC(M0)'))
dev.off()


#####[Table 2: modelos en N+1]#####################################################################################################
# M0 Baseline model
M0_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M1 Baseline model + Cloze
M1_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps +
             CLOZE_pred + CLOZE_pred_next +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M2 Baseline model + ngramCache
M2_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M3 Baseline model + LSA009
M3_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             LSA009.promedio.conSW + LSA009.promedio.conSW_next +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M4 Baseline model + ngramCahche + LSA 
M4_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M4 Baseline model + ngramCahche + LSA 
M5_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             LSA009.promedio.conSW + LSA009.promedio.conSW_next + 
             X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M6 Baseline model + ngramCahche + FT50
M6_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
             X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M7 Baseline model + LSA + FT50
M7_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
             LSA009.promedio.conSW + LSA009.promedio.conSW_next + 
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M8 Baseline model + ngramCahche + FT50 + LSA15
M8_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
             LSA009.promedio.conSW + LSA009.promedio.conSW_next +
             X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

# M9 Baseline model + ngramCahche (N) + FT50 (N+1)
M9_N1 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
             rpl + rpt + rps + 
             X4.gramcache.0.0001500000_0.15 +
             FT050.distancia.promedio_conSW_wiki_next +
             (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

CI_M0_N1<- confint(M0_N1)
CI_M1_N1<- confint(M1_N1)
CI_M2_N1<- confint(M2_N1)
CI_M3_N1<- confint(M3_N1)
CI_M4_N1<- confint(M4_N1)
CI_M5_N1<- confint(M5_N1)
CI_M6_N1<- confint(M6_N1)
CI_M7_N1<- confint(M7_N1)
CI_M8_N1<- confint(M8_N1)
CI_M9_N1<- confint(M9_N1)

anovas_N1 <- data.frame("M0.N+1" = c(NA,NA),
                     "M1.N+1" = c(anova(M0_N1,M1_N1)["M1_N1","Pr(>Chisq)"], NA),
                     "M2.N+1" = c(anova(M0_N1,M2_N1)["M2_N1","Pr(>Chisq)"], anova(M1_N1,M2_N1)["M2_N1","Pr(>Chisq)"]),
                     "M3.N+1" = c(anova(M0_N1,M3_N1)["M3_N1","Pr(>Chisq)"], anova(M1_N1,M3_N1)["M3_N1","Pr(>Chisq)"]),
                     "M4.N+1" = c(anova(M0_N1,M4_N1)["M4_N1","Pr(>Chisq)"], anova(M1_N1,M4_N1)["M4_N1","Pr(>Chisq)"]),
                     "M5.N+1" = c(anova(M0_N1,M5_N1)["M5_N1","Pr(>Chisq)"], anova(M1_N1,M5_N1)["M5_N1","Pr(>Chisq)"]),
                     "M6.N+1" = c(anova(M0_N1,M6_N1)["M6_N1","Pr(>Chisq)"], anova(M1_N1,M6_N1)["M6_N1","Pr(>Chisq)"]),
                     "M7.N+1" = c(anova(M0_N1,M7_N1)["M7_N1","Pr(>Chisq)"], anova(M1_N1,M7_N1)["M7_N1","Pr(>Chisq)"]),
                     "M8.N+1" = c(anova(M0_N1,M8_N1)["M8_N1","Pr(>Chisq)"], anova(M1_N1,M8_N1)["M8_N1","Pr(>Chisq)"]),
                     "M9.N+1" = c(anova(M0_N1,M9_N1)["M9_N1","Pr(>Chisq)"], anova(M1_N1,M9_N1)["M9_N1","Pr(>Chisq)"]))
row.names(anovas_N1) <- c("M0.N+1","M1.N+1")
corrplot(as.matrix(anovas_N1), is.corr = FALSE, method = "color", addgrid.col = TRUE,
         addCoef.col = TRUE,
         na.label = '-',
         col=colorRampPalette(c("blue","white","red"))(200),
         tl.col = 'black',tl.srt= 0)   

df <- data.frame(effect=c(''))
df_IC_N1 <- data.frame(effect=c(''))
for (i in c(0:9)){
  modelName <- paste0('M',i,'_N1')
  
  tmp <- eval(parse(text=modelName))
  lst <- coef(summary(tmp))[,"t value"]
  lst <- data.frame(effect=names(lst), lst)
  colnames(lst)[ncol(lst)] <- modelName
  
  df <- merge(df, lst, by = 'effect', all = TRUE)
  
  # estimates
  es <- coef(summary(tmp))[,"Estimate"]
  es <- data.frame(effect=names(es), es)
  colnames(es)[ncol(es)] <- paste0(modelName, "_b")
  
  # intervalos de confianza
  CIName <- paste0('CI_M',i,'_N1')
  tmp <- eval(parse(text=CIName))
  ci <- data.frame(effect=row.names(tmp), tmp)
  colnames(ci) <- c("effect", paste0(modelName, "_2.5"), paste0(modelName, "_97.5"))
  ci <- ci[5:dim(ci)[1],]
  
  ci <- merge(es,ci, by = 'effect', all = TRUE)
  df_IC_N1 <- merge(df_IC_N1, ci, by = 'effect', all = TRUE)
}
rownames(df) <- df[,1]
df[,1] <- NULL 
rownames(df_IC_N1) <- df_IC_N1[,1]
df_IC_N1[,1] <- NULL 
df_IC_N1 <- df_IC_N1[-1,]# La primera fila esta vacia por construccion
format(df_IC_N1, digits=1)
write.csv(df_IC_N1, paste0(figFolder,'S3-IC_N1.csv'))

# Corro los modelos remef y los voy agregando al df
for (i in c(2:9)){
  modelName <- paste0('M',i,'_N1')
  tmp <- eval(parse(text=modelName))
  
  fixef = rownames(coef(summary(tmp)))
  data$remefData <- remef(tmp, fix = fixef)
  Mr <- lmer(remefData ~ CLOZE_pred + CLOZE_pred_next +
             + (1|sujid) + (1|textid) + (1|wordid), 
             data = data, REML = FALSE)
  
  lst <- coef(summary(Mr))[,"t value"]
  df['BORRAR',modelName] <- NA
  df['CLOZE_pred_remef',modelName] <- lst['CLOZE_pred']
  df['CLOZE_pred_next_remef',modelName] <- lst['CLOZE_pred_next']
}
df <- df[-1,]# La primera fila esta vacia por construccion
df <- df[!rownames(df) %in% '(Intercept)',] # No me interesa mostrar el intercept (me saca de rango los colores)

df <- data.matrix(df)

pdf(paste0(figFolder, "tmp.pdf"), width=12,height = 13)
#pdf(paste0(figFolder, "4-TablaModelosN1.pdf"), width=13,height = 13)
corrplot(df, is.corr = FALSE, method = "color", 
         addgrid.col = TRUE, addCoef.col = TRUE,
         na.label = '-',
         col=colorRampPalette(c("blue","white","red"))(200),
         tl.col = 'black',tl.srt= 0, tl.offset = 1,
         number.cex = .8)
dev.off()


# Grafico AICs de los modelos 1-7
AICm0 = extractAIC(M0_N1)[2]
aics = c(extractAIC(M1_N1)[2],
         extractAIC(M2_N1)[2],
         extractAIC(M3_N1)[2],
         extractAIC(M4_N1)[2],
         extractAIC(M5_N1)[2],
         extractAIC(M6_N1)[2],
         extractAIC(M7_N1)[2],
         extractAIC(M8_N1)[2],
         extractAIC(M9_N1)[2]) - AICm0

pdf(paste0(figFolder, "4-AICModelosN1.pdf"))
barplot(aics, ylim = c(min(aics)-100, max(aics)+100),xpd = FALSE, 
        ylab = c('AIC(Mi) - AIC(M0)'))
dev.off()


#####[Box 3.1: optimizo ngram en n+1]###################################################################################################
#Optimizo N en N+1
ngrams2 <- read.csv('~/Documentos/Repos/ngrams/TABLE2')
enes=1:7
correlation = c()
for (n in enes){
  name=paste0('X',n,'.gram')
  correlation = c(correlation,cor(ngrams2$cloze_predictor, ngrams2[name]))
}
df = data.frame(enes, correlation)

ggplot(df, aes(x = enes, y = correlation, colour = 1)) +
  geom_point() +
  geom_line()


ggplot(df, aes(x = enes, y = correlaciones, color))

# optimizo Delta y lambda N+1
columnas  <- colnames(data) 
fields    <- columnas[grepl('X4.gram_nextcache.*_next',columnas)]

aicngram <- tvalngram <- deltas <- lambdas  <- c()
for (f in fields){
  print(f)
  
  # Extraigo el delta y el lambda del nombre
  todoJunto <- strsplit(f,'X4.gram_nextcache.')[[1]][2]
  separado  <- strsplit(todoJunto,'_')[[1]]
  deltas    = c(deltas,  separado[1])
  lambdas   = c(lambdas, separado[3])  
  
  # w2v promedio
  str = 'ngram'
  model = modelo(str, f,data)
  aicngram  = c(aicngram, model[1])
  tvalngram = c(tvalngram, model[2])
}

df <- data.frame(aicngram, tvalngram, deltas,lambdas)
# save(df, file='Optim_delta_lambda2.rda')
# load(file='Optim_delta_lambda2.rda')

length(aicngram)
d <- sort(unique(df$deltas))
l <- sort(unique(df$lambdas))

AICm  <- matrix(data=NA,nrow=length(d),ncol=length(l))
tvalm <- matrix(data=NA,nrow=length(d),ncol=length(l))
colnames(AICm) <- colnames(tvalm) <- l
rownames(AICm) <- rownames(tvalm) <- d

for (it in 1:length(df$aicngram)){
  i = df$deltas[it]
  j = df$lambdas[it]
  
  AICm[i,j]  <- df$aicngram[it]  
  tvalm[i,j] <- df$tvalngram[it]
}

AICm <- t(AICm)
levelplot(AICm,
          xlab = 'lambda',
          ylab = 'delta',
          col.regions = colorRampPalette(brewer.pal(9,"Reds"))(16))

inds = which(AICm == min(AICm), arr.ind = T)
colnames(AICm)[inds[2]]
rownames(AICm)[inds[1]]


#####[Diagnosis]###################################################################################################

M_FULL <- lmer(logFPRT ~ Nlaunchsite + invlength*freq + 
                rpl + rpt + rps + 
                FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
                LSA009.promedio.conSW + LSA009.promedio.conSW_next +
                X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 + 
                CLOZE_pred + CLOZE_pred_next +
                (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)

n = names(coef(summary(M_FULL))[,"t value"])
n = n[2:(length(n)-1)]

# Residual Plot
plot(M_FULL)
for (i in c(0:8)){
  print(i)
  modelName <- paste0('M',i,'_N')
  tmp <- eval(parse(text=modelName))
  modelName <- gsub("_N",".N",modelName)
  p <- plot(tmp, main = modelName, xlab = "", ylab = "")
  nam <- paste("A", i, sep = "")
  assign(nam, p)
  }
grid.arrange(A0, A1, A2, A3, A4, A5, A6, A7, A8)


for (i in c(1:9)){
  modelName <- paste0('M',i,'_N1')
  tmp <- eval(parse(text=modelName))
  modelName <- gsub("_N",".N+",modelName)
  p <- plot(tmp, main = modelName, xlab = "", ylab = "")
  nam <- paste("A", i, sep = "")
  assign(nam, p)
  }
grid.arrange(A1, A2, A3, A4, A5, A6, A7, A8,A9)

# Linearity in each variable
for(i in 1:length(n)){
  nam <- paste("A", i, sep = "")
  p <- ggplot(data.frame(fixed=data[,n[i]],pearson=residuals(M_FULL,type="pearson")),
              aes(x=fixed,y=pearson)) +xlab(n[i])+
    geom_point() +
    theme_bw()
  assign(nam, p)
}
grid.arrange(A1, A2, A3, A4, A5, A6, A7, A9, A10, A11, A12, A13, A14)   

# Normality of residuals
par(mfrow=c(3,3))
for (i in c(0:8)){
  modelName <- paste0('M',i,'_N')
  tmp <- eval(parse(text=modelName))
  modelName <- gsub("_N",".N",modelName)
  qqnorm(residuals(tmp), main = modelName, xlab = "", ylab = "")
}

par(mfrow=c(3,3))
for (i in c(1:9)){
  modelName <- paste0('M',i,'_N1')
  tmp <- eval(parse(text=modelName))
  modelName <- gsub("_N",".N+",modelName)
  qqnorm(residuals(tmp), main = modelName, xlab = "", ylab = "")
}



# Independence wordid
LM_FULL <- lm(logFPRT ~ Nlaunchsite + invlength*freq + 
                rpl + rpt + rps + 
                FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
                LSA009.promedio.conSW + LSA009.promedio.conSW_next +
                X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 + 
                CLOZE_pred + CLOZE_pred_next +
                wordid, data = data)
lmcoefs <- summary(LM_FULL)$coefficients[,"Estimate"]

means <- aggregate(data[,names(lmcoefs)[2:15]],by=list(data$wordid),FUN=mean)
means$effects <- c(0,lmcoefs[substr(names(lmcoefs),1,6) == "wordid"])
means$effects <- means$effects - mean(means$effects)

cor(means[,c(names(lmcoefs)[2:15], "effects")])

# Independence TEXTID

LM_FULL <- lm(logFPRT ~ Nlaunchsite + invlength*freq + 
                rpl + rpt + rps + 
                FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
                LSA009.promedio.conSW + LSA009.promedio.conSW_next +
                X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 + 
                CLOZE_pred + CLOZE_pred_next +
                textid, data = data)
lmcoefs <- summary(LM_FULL)$coefficients[,"Estimate"]

means <- aggregate(data[,names(lmcoefs)[2:15]],by=list(data$textid),FUN=mean)
means$effects <- c(0,lmcoefs[substr(names(lmcoefs),1,6) == "textid"])
means$effects <- means$effects - mean(means$effects)

cor(means[,c(names(lmcoefs)[2:15], "effects")])

# Independence SUJID

LM_FULL <- lm(logFPRT ~ Nlaunchsite + invlength*freq + 
                rpl + rpt + rps + 
                FT050.distancia.promedio_conSW_wiki + FT050.distancia.promedio_conSW_wiki_next +
                LSA009.promedio.conSW + LSA009.promedio.conSW_next +
                X4.gramcache.0.0001500000_0.15 + X4.gram_nextcache.0.0001500000_next_0.15 + 
                CLOZE_pred + CLOZE_pred_next +
                sujid, data = data)
lmcoefs <- summary(LM_FULL)$coefficients[,"Estimate"]

means <- aggregate(data[,names(lmcoefs)[2:15]],by=list(data$sujid),FUN=mean)
means$effects <- c(0,lmcoefs[substr(names(lmcoefs),1,5) == "sujid"])
means$effects <- means$effects - mean(means$effects)

cor(means[,c(names(lmcoefs)[2:15], "effects")])

