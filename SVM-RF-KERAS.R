library(randomForest)
library(ROCR)
library(e1071)
library(keras)
library(tensorflow)

#sample
ntrain = sample(nrow(all.judge),floor(0.7*nrow(all.judge)),replace=FALSE)
train = all.judge[ntrain,]
test = all.judge[-ntrain,]
test$judge <- as.factor(test$judge)
test.x <- as.matrix(test[,3:1278])
train <- train[,-which(names(train)=="mcq240y")]
train <- train[,-which(names(train)=="mcq240r")]
train <- train[,-which(names(train)=="mcq240k")]
train <- train[,-which(names(train)=="mcq240i")]
train <- train[,-which(names(train)=="mcq240d")]
train <- train[,-which(names(train)=="mcq230d")]
test <- test[,-which(names(test)=="mcq240y")]
test <- test[,-which(names(test)=="mcq240r")]
test <- test[,-which(names(test)=="mcq240k")]
test <- test[,-which(names(test)=="mcq240i")]
test <- test[,-which(names(test)=="mcq240d")]
test <- test[,-which(names(test)=="mcq230d")]

#randomForest
rf.train.x <- as.matrix(train[,2:1278])
rf.train.x <- as.matrix(train[,3:1278])
rf.train.y <- as.matrix(train[,2])
rf.test.x <- as.matrix(test[,3:1278])
rf.test.y <- as.matrix(test[,2])

rf.model <- randomForest(x=rf.train.x,y=rf.train.y,ntree=500,mtry=600,importance=TRUE,proximity=TRUE)
print(rf.model)
rf.predict <- predict(rf.model,rf.test.x)
rf.predict <- ifelse(rf.predict>0.5,1,0)
rf.prediction <- prediction(rf.predict,rf.test.y)
rf.prediction.auc <- performance(rf.prediction,'auc')
rf.prediction.auc@y.values
[[1]]
[1] 0.6565282

#SVM
test$judge <- as.factor(test$judge)
svm.train <- as.matrix(train[,2:1278])
svm.test.x <- as.matrix(test[,3:1278])
svm.test.y <- as.matrix(test[,2])

svm.model <- svm(judge~.,data=svm.train,type="C",kernel="radial")
svm.pre <- predict(svm.model,newdata = svm.test.x)
svm.prediction <- prediction(svm.pre,svm.test.y)
svm.prediction.auc <- performance(svm.prediction,'auc')
svm.prediction.auc@y.values
[[1]]
[1] 0.9559798

#Keras
keras.train.x <- as.matrix(train[,3:1278])
keras.train.labels <- as.matrix(train[,2])
keras.test.x <- as.matrix(test[,3:1278])
keras.test.labels <- as.matrix(test[,2])

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = c(1276)) %>%
  layer_dropout(rate = 0.01) %>% 
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.01) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile( 
    optimizer = optimizer_rmsprop(),
    loss = loss_binary_crossentropy,
    metrics = c('accuracy')
  )

history <- model %>% fit(keras.train,keras.train.labels, epochs=30, batch_size=64)
keras.pre <- predict(model,keras.test)

#GLM
SRK.data <- as.data.frame(cbind(test$judge,svm.pre,rf.predict,keras.pre))
names(SRK.data) = c("judge","svm.pre","rf.pre","keras.pre")
SRK.data$judge <- ifelse(SRK.data$judge==2,1,0)

SRK.judgePredict <-predict(SRK.model,test.x)
predict <- prediction(SRK.judgePredict,test$judge)
predict.auc <- performance(predict,'auc')
predict.auc@y.values
[[1]]
[1] 0.7093018


