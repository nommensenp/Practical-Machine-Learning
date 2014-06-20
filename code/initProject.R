#
#
#
library(caret)

data <-  read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

# remove columns that are filled mainly with NA's
countNA <- function(x) (sum(is.na(x)))
noNA <- (sapply(data,countNA)==0)
data <- data[,noNA]

# split data in training-testing and validation sets
set.seed(43563)
intrain <- createDataPartition(y = data$classe, p=0.6, list=FALSE)
training <- data[intrain,]
testing <-  data[-intrain,]

invalidate <- createDataPartition(y = testing$classe, p=0.5, list = FALSE)
validate   <- testing[invalidate,]
testing    <- testing[-invalidate,]

