Project Write-up for the course 
"Practical Machine Learning" (Jul 07 2014) project
========================================================

Background
========================================================
The project task includes qualitative excercise classification.  

 

The WLE(Weight Lifting Exercises) dataset is made available by the authors Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. of the paper "Qualitative Activity Recognition of Weight Lifting Exercises", Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

The data is collected by using several tracking devices attached to the subjects. The subjects performed correct and incorrect weight lifting movements. The class feature in the dataset indicates the quality of the movement, with 5 classes, where only A indicates the correct movement.


Cleaning the data
========================================================

At first the data is loaded, empty or NA strings are replaced with R NA.
    

```r
badEntries<-c("","NA", "#DIV/0!")
traindata <- read.csv("pml-training.csv", na.strings=badEntries)
testdata <- read.csv("pml-testing.csv", na.strings=badEntries)
```


```r
dim(traindata)
```

```
## [1] 19622   160
```
The data that can be used for training is 19622 observations with 159 features and one reponse variable.

We analyse how prevalent are the NA values in the training data.

```r
countVectNA<- function(x) {sum(is.na(x))}
columnNA <- apply(traindata,2,countVectNA)
keepColumns<-(columnNA==0)
```


Since there are 160 features an efficient way to get an overview would be a simple plot.


```r
plot(columnNA, xlab="Feature index", ylab="Count of NA values" )
```

![plot of chunk Empty columns plot](figure/Empty columns plot.png) 

We can see that the data is either provided or entirelly missing, so we do not have to deal with the missing data using methods as e.g. knnImpute, we just remove the empty columns. Also since this is not a time series analysis project, we are removing the time - related data.


```r
traindata <- traindata[,keepColumns]
testdata <- testdata[,keepColumns]
not_interesting<-c(1,2,3,4,5,6,7)
traindata<-traindata[,-not_interesting]
testdata<-testdata[,-not_interesting]
```



Feature reduction
=========================================================

```r
columncount<-dim(traindata)[2]
columncount
```

```
## [1] 53
```

We are dealing with 52 features, which is computationally feasible. Nevertheless it would be interesting to see if features can be reduced and satisfactory accuracy can be achieved. We do so by looking at features that have some corelation to the class.



```r
delme<-data.frame(traindata[,-columncount], as.numeric(traindata[,columncount]))
corclass<-cor(delme)[,columncount]
corelated<-c(which(corclass > 0.05))
traindata<-traindata[,corelated]
testdata<-testdata[,corelated]
```



This leaves us with the following 12 features

```r
names(traindata)
```

```
##  [1] "roll_belt"           "total_accel_belt"    "roll_arm"           
##  [4] "accel_arm_x"         "magnet_arm_x"        "pitch_dumbbell"     
##  [7] "accel_dumbbell_x"    "accel_dumbbell_z"    "magnet_dumbbell_x"  
## [10] "magnet_dumbbell_z"   "pitch_forearm"       "total_accel_forearm"
## [13] "classe"
```

Training and testing
===========================================================
Since we cannot tune the model on testdata (only two submissions are allowed), we will be taking a part of the training dataset for testing. Using a part of the training data as testdata also reduces the out of sample error, which can be important in some algorithms(linear/logistic regression), less so in others(random forests). 


```r
library(caret)
set.seed(314568)
train_index <- createDataPartition(y = traindata$classe, p=0.7,list=FALSE) 
mtrain<-traindata[train_index,]
mtest<-traindata[-train_index,]
```

We first look at the graphical information of a random subset of the data. 


```r
viz_index <- createDataPartition(y = mtrain$classe, p=0.1,list=FALSE) 
pairs(mtrain[viz_index,c(1:9, 13)])
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7.png) 

We can identify that data is clustered, but some parameters are corelated. Random forest with principal component analysis would be a suitable machine learning algorithm for such data.


```r
forest1<-randomForest(classe ~., data=mtrain, method="pca")
predicted<-predict(forest1, newdata=mtest[,-13])
confusionMatrix(mtest$classe, predicted)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1653    5    8    6    2
##          B   24 1081   29    2    3
##          C    0   16  996   14    0
##          D    3    0   25  932    4
##          E    0    9    4    8 1061
## 
## Overall Statistics
##                                         
##                Accuracy : 0.972         
##                  95% CI : (0.968, 0.977)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.965         
##  Mcnemar's Test P-Value : 1.31e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.984    0.973    0.938    0.969    0.992
## Specificity             0.995    0.988    0.994    0.993    0.996
## Pos Pred Value          0.987    0.949    0.971    0.967    0.981
## Neg Pred Value          0.994    0.994    0.986    0.994    0.998
## Prevalence              0.285    0.189    0.180    0.163    0.182
## Detection Rate          0.281    0.184    0.169    0.158    0.180
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.989    0.980    0.966    0.981    0.994
```



```r
forest2<-randomForest(classe ~., data=mtrain, method="center, scale")
predicted<-predict(forest2, newdata=mtest[,-13])
confusionMatrix(mtest$classe, predicted)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1657    4    7    4    2
##          B   25 1081   29    2    2
##          C    0   18  993   14    1
##          D    3    0   27  930    4
##          E    0   11    3    8 1060
## 
## Overall Statistics
##                                         
##                Accuracy : 0.972         
##                  95% CI : (0.968, 0.976)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.965         
##  Mcnemar's Test P-Value : 8.8e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.983    0.970    0.938    0.971    0.992
## Specificity             0.996    0.988    0.993    0.993    0.995
## Pos Pred Value          0.990    0.949    0.968    0.965    0.980
## Neg Pred Value          0.993    0.993    0.986    0.994    0.998
## Prevalence              0.286    0.189    0.180    0.163    0.182
## Detection Rate          0.282    0.184    0.169    0.158    0.180
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.990    0.979    0.965    0.982    0.994
```
With forest2 it appears the PCA is not essential (the algorithm performs slightly better without it) so only center and scale preprocessing methods are usef for the final prediction. 
