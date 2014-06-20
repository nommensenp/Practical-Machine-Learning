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
## 11776 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.04         0.06    
##   0.06  0.4       0.2    0.06         0.09    
##   0.1   0.3       0.08   0.04         0.06    
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
## Prediction    A    B    C    D    E
##          A 1018  312  312  284   95
##          B   23  277   21  113   96
##          C   75  170  351  246  209
##          D    0    0    0    0    0
##          E    0    0    0    0  321
## 
## Overall Statistics
##                                         
##                Accuracy : 0.501         
##                  95% CI : (0.486, 0.517)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.349         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.912   0.3650   0.5132    0.000   0.4452
## Specificity             0.643   0.9200   0.7839    1.000   1.0000
## Pos Pred Value          0.504   0.5226   0.3340      NaN   1.0000
## Neg Pred Value          0.948   0.8579   0.8841    0.836   0.8890
## Prevalence              0.284   0.1935   0.1744    0.164   0.1838
## Detection Rate          0.259   0.0706   0.0895    0.000   0.0818
## Detection Prevalence    0.515   0.1351   0.2679    0.000   0.0818
## Balanced Accuracy       0.777   0.6425   0.6485    0.500   0.7226
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
##          A 701 143 168  42  64
##          B 125 350  62 111 133
##          C 115 121 365 102  94
##          D 164  94  66 315  87
##          E  11  51  23  73 343
## 
## Overall Statistics
##                                         
##                Accuracy : 0.529         
##                  95% CI : (0.513, 0.544)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.404         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.628   0.4611    0.534   0.4899   0.4757
## Specificity             0.851   0.8638    0.867   0.8747   0.9507
## Pos Pred Value          0.627   0.4481    0.458   0.4339   0.6846
## Neg Pred Value          0.852   0.8698    0.898   0.8974   0.8895
## Prevalence              0.284   0.1935    0.174   0.1639   0.1838
## Detection Rate          0.179   0.0892    0.093   0.0803   0.0874
## Detection Prevalence    0.285   0.1991    0.203   0.1851   0.1277
## Balanced Accuracy       0.740   0.6625    0.700   0.6823   0.7132
```
### Predicting with a random forrest

```r
modRF <- train(classe ~ . , method = "rf", data=training, prox=TRUE)
print(modRF)
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.002   
##   30    1         1      0.002        0.003   
##   60    1         1      0.005        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
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
##          A 1115   11    0    0    0
##          B    1  743   14    0    0
##          C    0    5  669   15    0
##          D    0    0    1  626    1
##          E    0    0    0    2  720
## 
## Overall Statistics
##                                         
##                Accuracy : 0.987         
##                  95% CI : (0.983, 0.991)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.984         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.979    0.978    0.974    0.999
## Specificity             0.996    0.995    0.994    0.999    0.999
## Pos Pred Value          0.990    0.980    0.971    0.997    0.997
## Neg Pred Value          1.000    0.995    0.995    0.995    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.189    0.171    0.160    0.184
## Detection Prevalence    0.287    0.193    0.176    0.160    0.184
## Balanced Accuracy       0.998    0.987    0.986    0.986    0.999
```

Selecting the optimal model
---------------------------
Several algorithms were tested. 
* CART with a accuracy from cross validation of 0.5014.
* LDA with a accuracy from cross validation of 0.5287.
* Random forrest with a accuracy from cross validation of 0.9873.

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
##          A 1116    1    0    0    0
##          B    0  757    6    0    0
##          C    0    1  677   16    0
##          D    0    0    1  627    2
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.995)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.991        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.990    0.975    0.997
## Specificity             1.000    0.998    0.995    0.999    1.000
## Pos Pred Value          0.999    0.992    0.976    0.995    1.000
## Neg Pred Value          1.000    0.999    0.998    0.995    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.160    0.183
## Detection Prevalence    0.285    0.194    0.177    0.161    0.183
## Balanced Accuracy       1.000    0.998    0.992    0.987    0.999
```

__The accuracy on the validation set is 0.9931. This indicates the model works rather well.__ This number is also close the result of the cross validation.
