#
#
#
library(caret)

data <-  read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

# remove columns that are filled mainly with NA's
countNA <- function(x) (sum(is.na(x)))
noNA <- (sapply(data,countNA)==0)
data <- data[,noNA]
data <- data[,c(-1,-3, -4, -5, -6, -7)]

dataAns <- read.csv("data/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
dataAns <- dataAns[,noNA]
dataAns <- dataAns[,c(-1,-3, -4, -5, -6, -7)]

# split data in training-testing and validation sets
set.seed(43563)
intrain <- createDataPartition(y = data$classe, p=0.6, list=FALSE)
training <- data[intrain,]
testing <-  data[-intrain,]

invalidate <- createDataPartition(y = testing$classe, p=0.5, list = FALSE)
validate   <- testing[invalidate,]
testing    <- testing[-invalidate,]


inSub <- createDataPartition(y = training$classe, p=0.05, list=FALSE)
trainSub <- training[inSub,]

# train a model
modTree <- train(classe ~ . , met5hod = "rpart", data=training)
print(modTree)

prediction <- predict(modTree, newdata=testing) 
confusionMatrix(prediction, testing$classe)
qplot(testing$classe, prediction)

modLDA <- train(classe ~ . , method = "lda", preProcess="pca", data=training)
prediction <- predict(modLDA, newdata=testing) 
confusionMatrix(prediction, testing$classe)

modRF <- train(classe ~ . , method = "rf", data=training, prox=TRUE)
prediction <- predict(modRF, newdata=testing) 
confusionMatrix(prediction, testing$classe)


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

prediction <- predict(modRF, newdata=dataAns)
print(prediction)
pwd <- getwd()
setwd("projectAnwsers")
