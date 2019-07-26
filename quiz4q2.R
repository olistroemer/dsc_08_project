library(caret)
library(gbm)

set.seed(3433)

library(AppliedPredictiveModeling)

data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)

mrf  <- train(diagnosis ~ ., data=training, method="rf")
mgbm <- train(diagnosis ~ ., data=training, method="gbm")
mlda <- train(diagnosis ~ ., data=training, method="lda")

prf  <- predict(mrf,  newdata=testing)
pgbm <- predict(mgbm, newdata=testing)
plda <- predict(mlda, newdata=testing)

pDF  <- data.frame(prf, pgbm, plda, diagnosis=testing$diagnosis)

mc   <- train(diagnosis ~ ., data=pDF, method="rf")
pc   <- predict(mc, newdata=testing)

confusionMatrix(prf,  testing$diagnosis)
confusionMatrix(pgbm, testing$diagnosis)
confusionMatrix(plda, testing$diagnosis)
confusionMatrix(pc,   testing$diagnosis)
