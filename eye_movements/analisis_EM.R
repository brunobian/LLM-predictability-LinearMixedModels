# options(digits=2)
rm(list=ls()) #borro todas las variables del workspace (rm)

setwd('/home/freddy/Escritorio/proj/pasantia_LIAA/fastai_custom_awd_lstm/eye_movements')
#setwd('/media/brunobian/ExtraDrive1/Repos/awdlstm-cloze-task/eye_movements/')

library(lme4)
library(plyr)
library(dplyr)
library(ggplot2)
library(lattice)
library(RColorBrewer)
library(reshape2)
library(corrplot)
library(MuMIn)


##### Load DataFrame2019 and apply filter #####################################################################################################
# Load Eye data
# Esto viene de la ultima version del paper de LSA. 
# load("DataFilt.rda")
# f <- colnames(data)
# ind <- c(grep(".gram",f), grep("LSA",f), grep("w2v",f), grep("FT",f), grep("cache",f))
# data2 <- data[,-ind]
# data2["X4.gramcache.0.0001500000_0.15"] <- data["X4.gramcache.0.0001500000_0.15"]
# data2["FT050.distancia.promedio_conSW_wiki"] <- data["FT050.distancia.promedio_conSW_wiki"]  
# data2["LSA009.promedio.conSW"] <- data["LSA009.promedio.conSW"]
# head(data2)
# data <- data2
# save(data, file="Data2020.rda")

##### Load DataFrame #####################################################################################################


# for (prefix in c('lstm', 'ft', '10_all', 'ft_maj', '10_all_maj')) {
for (prefix in c('ft_maj')) {

   print(paste("prefix ", prefix))

  load("Data2020.rda")

  dn <- read.csv(paste0('../resultados/', prefix, '/all.csv'))

  data <- merge(data, dn)
  head(data)

  ## Columnas importantes del dataframe completo
  # Ngram: X4.gramcache.0.0001500000_0.15
  # FastText: FT050.distancia.promedio_conSW_wiki
  # LSA: LSA009.promedio.conSW
  # AWD: awd

  cor(data$awd, data$CLOZE_pred, method="pearson")

  M_i <- function(data, extra) {
    print(paste("Doing Mi ", extra))
    m <- lmer(
      as.formula(
        paste(
          'logFPRT ~ Nlaunchsite + invlength * freq + ',
          'rpl + rpt + rps + ',
          extra,
          ' + (1 | sujid) + (1 | textid) + (1 | wordid)'
        )
      ), data = data, REML = FALSE
    )
    return(m)
  }


  M0 <- lmer(logFPRT ~ Nlaunchsite + invlength*freq +
               rpl + rpt + rps +
               (1|sujid) + (1|textid) + (1|wordid), data = data, REML = FALSE)
  #summary(M0, cor=FALSE)

  M1 <- M_i(data, 'CLOZE_pred')
  M2 <- M_i(data, 'X4.gramcache.0.0001500000_0.15')
  M3 <- M_i(data, 'LSA009.promedio.conSW')
  M4 <- M_i(data, 'FT050.distancia.promedio_conSW_wiki')
  M5 <- M_i(data, 'X4.gramcache.0.0001500000_0.15 + LSA009.promedio.conSW')
  M6 <- M_i(data, 'X4.gramcache.0.0001500000_0.15 + FT050.distancia.promedio_conSW_wiki')
  M7 <- M_i(data, 'LSA009.promedio.conSW + FT050.distancia.promedio_conSW_wiki')
  M8 <- M_i(data, 'X4.gramcache.0.0001500000_0.15 + LSA009.promedio.conSW + FT050.distancia.promedio_conSW_wiki')
  M9 <- M_i(data, 'awd')
  M10 <- M_i(data, 'awd + X4.gramcache.0.0001500000_0.15')
  M11 <- M_i(data, 'awd + LSA009.promedio.conSW')
  M12 <- M_i(data, 'awd + FT050.distancia.promedio_conSW_wiki')
  M13 <- M_i(data, 'awd + X4.gramcache.0.0001500000_0.15 + LSA009.promedio.conSW')
  M14 <- M_i(data, 'awd + X4.gramcache.0.0001500000_0.15 + FT050.distancia.promedio_conSW_wiki')
  M15 <- M_i(data, 'awd + LSA009.promedio.conSW + FT050.distancia.promedio_conSW_wiki')
  M16 <- M_i(data, 'awd + X4.gramcache.0.0001500000_0.15 + LSA009.promedio.conSW + FT050.distancia.promedio_conSW_wiki')

  source('remef.v0.6.10.R')
  #fixef <- rownames(coef(summary(M2)))
  #data$remefData <- remef(M2, fix = fixef)
  #Mr <- lmer(remefData ~ CLOZE_pred
  #           + (1|sujid) + (1|textid) + (1|wordid),
  #           data = data, REML = FALSE)
  #summary(Mr, cor=FALSE)


  tvalues <- data.frame()
  aic <- data.frame()

  for (i in 0:16){
    #modelName <- paste0('M',i,'_N')
    modelName <- paste0('M',i)
    tmp <- eval(parse(text=modelName))

    coef_s_tmps <- coef(summary(tmp))
    fixef <- rownames(coef_s_tmps)

    data$remefData <- remef(tmp, fix = fixef)
    Mr <- lmer(remefData ~ CLOZE_pred
               + (1|sujid) + (1|textid) + (1|wordid),
               data = data, REML = FALSE)

    lst <- coef(summary(Mr))[,"t value"]
    tvalues['CLOZE_pred_remef', modelName] <- lst['CLOZE_pred']

    for (j in fixef) {
      tvalues[j, modelName] <- coef_s_tmps[j, 't value']
    }

    aic['aic', modelName] <- AIC(tmp)

    r2 <- r.squaredGLMM(tmp)
    tvalues['R2m',modelName] <- r2[, 'R2m']
    tvalues['R2c',modelName] <- r2[, 'R2c']
  }

  write.csv(tvalues, paste0('../resultados/',prefix, '/tvalues.csv'))
  write.csv(aic, paste0('../resultados/', prefix, '/aic.csv'))
}