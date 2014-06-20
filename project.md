Project to Practical Machine Learning
========================================================
Paul Nommensen, 20-06-2014

Introduction
------------

This work was done in the frame of the Coursera course Practical Machine learning.

Data was obtained from [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har).  6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Data was collected from accelerometers on the belt, forearm, arm, and dumbell. Goal of the project is to predict the way in which the barbell was lifted.


```r
library(caret)
```

Loading the data
----------------

```r
data <-  read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
```
Note that the error message #DIV>0! in the data file are treated as a NA.

Cleansing the data
------------------
Many columns contain mainly NA's. These columns are removed from the data set.

```r
countNA <- function(x) (sum(is.na(x)))
noNA <- (sapply(data,countNA)==0)
data <- data[,noNA]

dim(data)
```

```
## [1] 19622    60
```

Covariate selection
-------------------
Some of the parameters contain information that seems to be irrelvant. The paramter x presents the order in which the observation were done. Also the actual moment in time the exercise was performed can not be used as a predictor because when an obervation is done at a later time it is out of range. So the parameters X, raw_timestamp_part_1, raw_timestamp_part_2 and cvtd_timestamp are removed from the data set. 

The parameters new_window and num_window also seems to indicate the orderin which the observation were done. It is anticipated that these do not contain real information on the way a excersise was done. So both are removed from the data set.

Note that the same persons are present in the test-dataset. So the paramter name can still be used in the predictions.
 

```r
data <- data[,c(-1,-3, -4, -5, -6, -7)]
```

Slicing the data
----------------
The data set is splitted in a training, test and validation set. The sets contain 60%, 20% and 20% of the data.

```r
set.seed(43563)
intrain <- createDataPartition(y = data$classe, p=0.6, list=FALSE)
training <- data[intrain,]
testing <-  data[-intrain,]

invalidate <- createDataPartition(y = testing$classe, p=0.5, list = FALSE)
validate   <- testing[invalidate,]
testing    <- testing[-invalidate,]

inSub  <- createDataPartition(y = training$classe, p=0.05, list=FALSE)
training <- training[inSub,]
```

Prepare a prediction model
--------------------------
### Predicting with a classification tree

```r
modTree <- train(classe ~ . , method = "rpart", data=training)
print(modTree)
```

```
## CART 
## 
## 591 samples
##  53 predictors
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 591, 591, 591, 591, 591, 591, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.06         0.07    
##   0.07  0.4       0.2    0.08         0.1     
##   0.1   0.3       0.04   0.04         0.05    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04.
```


```r
prediction <- predict(modTree, newdata=testing) 
c<-confusionMatrix(prediction, testing$classe)
accTree <- c$overall[1]
print(c)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 992 392 216 361 107
##          B  63 219  51  46  80
##          C  48 148 417 236 202
##          D   0   0   0   0   0
##          E  13   0   0   0 332
## 
## Overall Statistics
##                                         
##                Accuracy : 0.5           
##                  95% CI : (0.484, 0.515)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.346         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.889   0.2885    0.610    0.000   0.4605
## Specificity             0.617   0.9241    0.804    1.000   0.9959
## Pos Pred Value          0.480   0.4771    0.397      NaN   0.9623
## Neg Pred Value          0.933   0.8441    0.907    0.836   0.8913
## Prevalence              0.284   0.1935    0.174    0.164   0.1838
## Detection Rate          0.253   0.0558    0.106    0.000   0.0846
## Detection Prevalence    0.527   0.1170    0.268    0.000   0.0879
## Balanced Accuracy       0.753   0.6063    0.707    0.500   0.7282
```

### Predicting with Linear discriminant analysis
Another prediction algorithm is lda. In order to overcome the correlations among the predictors a PCA is performed on them and the PCA-scores are used in the modeling

```r
modLDA <- train(classe ~ . , method = "lda", preProcess="pca", data=training)
```


```r
prediction <- predict(modLDA, newdata=testing) 
c<-confusionMatrix(prediction, testing$classe)
accLDA <- c$overall[1]
print(c)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 655 165 203  44  77
##          B 129 276  65 117 108
##          C 134 117 323  70  86
##          D 175 120  70 337 108
##          E  23  81  23  75 342
## 
## Overall Statistics
##                                         
##                Accuracy : 0.493         
##                  95% CI : (0.477, 0.509)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.359         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.587   0.3636   0.4722   0.5241   0.4743
## Specificity             0.826   0.8676   0.8743   0.8558   0.9369
## Pos Pred Value          0.573   0.3971   0.4425   0.4160   0.6287
## Neg Pred Value          0.834   0.8504   0.8869   0.9017   0.8878
## Prevalence              0.284   0.1935   0.1744   0.1639   0.1838
## Detection Rate          0.167   0.0704   0.0823   0.0859   0.0872
## Detection Prevalence    0.292   0.1772   0.1861   0.2065   0.1387
## Balanced Accuracy       0.706   0.6156   0.6733   0.6899   0.7056
```
### Predicting with a random forrest

```r
modRF <- train(classe ~ . , method = "rf", data=training, prox=TRUE)
print(modRF)
```

```
## Random Forest 
## 
## 591 samples
##  53 predictors
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 591, 591, 591, 591, 591, 591, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.8       0.7    0.03         0.04    
##   30    0.8       0.7    0.03         0.03    
##   60    0.8       0.7    0.03         0.03    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```


```r
prediction <- predict(modRF, newdata=testing) 
c<-confusionMatrix(prediction, testing$classe)
accRF <- c$overall[1]
print(c)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1057   69    3    7    1
##          B   18  611   50    7   20
##          C    5   55  592   75   48
##          D   32   22   37  509   22
##          E    4    2    2   45  630
## 
## Overall Statistics
##                                         
##                Accuracy : 0.866         
##                  95% CI : (0.855, 0.877)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.831         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.947    0.805    0.865    0.792    0.874
## Specificity             0.971    0.970    0.944    0.966    0.983
## Pos Pred Value          0.930    0.865    0.764    0.818    0.922
## Neg Pred Value          0.979    0.954    0.971    0.959    0.972
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.269    0.156    0.151    0.130    0.161
## Detection Prevalence    0.290    0.180    0.198    0.159    0.174
## Balanced Accuracy       0.959    0.887    0.904    0.879    0.929
```

Selecting the optimal model
---------------------------
Several algorithms were tested. 
* CART with a accuracy from cross validation of 0.4996.
* LDA with a accuracy from cross validation of 0.4927.
* Random forrest with a accuracy from cross validation of 0.8664.

Based on the accuracy in predicting the test set,  the random forest model is selected as the optimal model. 


The out of sample Error
----------------------
The randomforest model is used to predicted a validation set.


```r
prediction <- predict(modRF, newdata=validate) 
c<-confusionMatrix(prediction, testing$classe)
accVal <- c$overall[1]
print(c)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1026   74    4    6    2
##          B   19  610   47    7   35
##          C   19   56  593   62   44
##          D   41   17   39  506   16
##          E   11    2    1   62  624
## 
## Overall Statistics
##                                         
##                Accuracy : 0.856         
##                  95% CI : (0.845, 0.867)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.818         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.919    0.804    0.867    0.787    0.865
## Specificity             0.969    0.966    0.944    0.966    0.976
## Pos Pred Value          0.923    0.850    0.766    0.817    0.891
## Neg Pred Value          0.968    0.954    0.971    0.959    0.970
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.262    0.155    0.151    0.129    0.159
## Detection Prevalence    0.283    0.183    0.197    0.158    0.178
## Balanced Accuracy       0.944    0.885    0.906    0.876    0.921
```

__The accuracy on the validation set is 0.8562. This is close to 1 indicating that the model predicts well.__
