Project Write-up for the course 
"Practical Machine Learning" (Jul 07 2014) project
========================================================

1. Cleaning the data

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
plot(columnNA)
```

![plot of chunk Empty columns plot](figure/Empty columns plot.png) 

preptrain <- traindata[,keepColumns]
preptest <- testdata[,keepColumns]


# remove the time data
not_interesting<-c(1,2,3,4,5,6,7)
preptest<-preptest[,-not_interesting]
preptrain<-preptrain[,-not_interesting]
corclass<-cor(preptest)[,53]
#corclass<-cors[,53]
important<-c(which(corclass > 0.15))
preptrain<-preptrain[,important]
preptest<-preptest[,important]

#Split the data into train and test
train_index <- createDataPartition(y = preptrain$classe, p=0.7,list=FALSE) # 3927 rows7
mtrain<-preptrain[train_index,]
mtest<-preptrain[-train_index,]
#set.seed(314568)
#library(caret)
#modLin<-train(classe~., method="glm", data = mtrain, family = binomial(link="logit"), preprocess=c("center", "scale"))
#modRfCenterSc<-train(classe~.,method = "rf", data = mtrain,preprocess=c("center", "scale"))
#modRfPca<-train(classe~.,method = "rf", data = mtrain,preprocess="pca")
#forest1<-randomForest(classe ~., data=mtrain, method="pca")
#preds<-predict(modFit, newdata=preptest)
#confusionMatrix(mtest$classe, preds)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
#pml_write_files(preds)




