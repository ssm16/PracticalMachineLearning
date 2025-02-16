# Practical Machine Learning: Classifying how well an activity was done


```r
knitr::opts_chunk$set(cache=TRUE)
```

### Introduction to the assignment
The task in this assignment is to analyse personal activity data taken by sensors
during a weight lifting exercise in order to predict how well the exercise was
done. To create the data set 6 participants performed barbell lifts correctly 
(class A) and incorrectly with different deviations in movement (classes B-F). 
The data comes from accelerometers on the belt, forearm and arm of all participants
as well as from the dumbbell. Further information can be found here:
http://groupware.les.inf.puc-rio.br/har

### Loading the data
To start the analyis, the training data is loaded:

```r
library(lubridate)

if (!file.exists("pml-training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                      "pml-training.csv")
        print(now())
  }
```

```
## [1] "2017-05-05 19:36:31 CEST"
```

```r
training_data<-read.csv("pml-training.csv", header=TRUE)
```

It has the following dimensions:

```r
dim(training_data)
```

```
## [1] 19622   160
```

For later use the testing data is also loaded:

```r
if (!file.exists("pml-testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                      "pml-testing.csv")
        print(now())
  }
```

```
## [1] "2017-05-05 19:36:33 CEST"
```

```r
testing_data<-read.csv("pml-testing.csv", header=TRUE)
```

We will not be looking at the testing data until the analysis is finished. 

### Separation of training data in training and validation data
In this assignment I use random sampling as cross validation, since I want to include
activitiy data of all of the participants in the training set which would not 
necessarily be the case in using k-fold. For each run 75% of the (training) data 
will be used for training, the other 25% will be used for validation. The classification
of the activity is stored in the "classe" variable.


```r
library(caret)
set.seed(1011)
inTrain<-createDataPartition(y=training_data$classe, p=0.75, list=FALSE)
training<-training_data[inTrain,]
validate<-training_data[-inTrain,]
```

### Prepration of the training data
According to the assignment any of the variables of the data set can be used for
the prediction of the classe variable. However, the training data contains several
numbers of variables which contain (almost) no information. In addition the data set
contains administrative columns (no. of trial, participant, timestamps,...) which are
not directly connected to the exercise and therefore excluded from learning.

These columns are deleted:

```r
# Deletion of "empty" columns
percentageNA<-round(apply(is.na(training),2,sum)/dim(training)[1], digits=2)
training<-training[percentageNA<0.1]
percentageEmpty<-round(apply(training=="",2,sum)/dim(training)[1], digits=2)
training<-training[percentageEmpty<0.1]

# Deletion of administrative columns
training<-training[8:60]
```
Note: a similar result (except of for the administrative columns) will also be given by nearZeroVars.

This preprocessing is now also done for the validation data:

```r
validate<-validate[percentageNA<0.1]
validate<-validate[percentageEmpty<0.1]
validate<-validate[8:60]
```

### Selection of the predictors and an adequate model
The predictors as well as the prediction model were chosen using cross validation. For feature
extraction I used the principal components analysis (PCA) which showed good separation of the data in
5 clusters.


```r
library(stats)
# calculation of PCA
training_pca <- prcomp(training[,1:52], scale = TRUE)

biplot(training_pca,scale=0)
```

![](QualityPrediction_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# percentage of explained variance
percent_var <- training_pca$sdev^2/sum(training_pca$sdev^2)

# Scree plot
plot(percent_var, type="b", xlab = "Numbers of principal component (PC)",
     ylab = "Percentage of explained variance by PC")
```

![](QualityPrediction_files/figure-html/unnamed-chunk-7-2.png)<!-- -->

```r
# cumulated variance
cumulated_var <- cumsum(training_pca$sdev^2)/sum(training_pca$sdev^2)
plot(cumulated_var)
```

![](QualityPrediction_files/figure-html/unnamed-chunk-7-3.png)<!-- -->

I also tried PCA only with uncorrelated columns but the result was worse than using all columns, so
I dropped this approach.

Based on the scree plot and the variances explained by the principal components, I used 6 and 18 principal 
components for training the models. 6 components because there was a change in the scree plot around 5 or 6 
principal components (depending on the training set) and 18 compontents because the explain about 90% of the
variance. According to the scree plot 2 principal components would also be an option, but since they only 
explain about 30% of the variance, I dropped this option. My analysis showed that (as expected) prediction 
using 18 principal components gave clearly better results.

### Training the model with the training data set and prediction with validation data
In my analysis I used the methods rpart (recursive partitioning and regression trees) and rf (random forest). For each
method I used 5 iterations using random sampling. Random forest gave steady results with a prediction accuracy of 97%.



```r
# create training set from classe variable and principal components
train_pca<-data.frame(classe = training$classe, training_pca$x)

# in this case we use the classe variable and 18 PCs
train_pca <- train_pca[,1:19]

# training of random forest model
modFit <- train(classe ~ .,data = train_pca, method = "rf")

# before actual prediction, validation data needs to be transformed according to principle
# components from training data
test <- predict(training_pca, newdata = validate)
print(str(test))
```

```
##  num [1:4904, 1:52] 4.13 4.09 4.1 4.11 4.14 ...
##  - attr(*, "dimnames")=List of 2
##   ..$ : chr [1:4904] "2" "5" "6" "11" ...
##   ..$ : chr [1:52] "PC1" "PC2" "PC3" "PC4" ...
## NULL
```

```r
test <- as.data.frame(test)

# in this case we use 18 PCs (see above), the test data has no classe variable
test <- test[,1:18]

# prediction of the classe variable from the test data with the calculated model from above
pred <- predict(modFit, test)
table(validate$classe,pred)
```

```
##    pred
##        A    B    C    D    E
##   A 1383    5    2    3    2
##   B   13  917   19    0    0
##   C    6    9  836    2    2
##   D    3    1   29  771    0
##   E    1    6    9    7  878
```

```r
# calculate and store accuracy for later use
sum(validate$classe==pred)/length(pred)
```

```
## [1] 0.9757341
```

### Prediction of testing data

In order to use the model to predict on the testing data, the testing data needs to
be prepared.

First the deleted columns from above are removed:

```r
testing<-testing_data[percentageNA<0.1]
testing<-testing[percentageEmpty<0.1]
testing<-testing[8:60]
```

As a next step, the principal components analysis from the training data is applied
to the testing data:

```r
test_pca<-predict(training_pca, newdata = testing)
test_pca<-as.data.frame(test_pca)

# again 18 principal components are used
test_pca<-test_pca[,1:18]
```

Now the prediction can be done:

```r
# prediction of the classe variable from the test data with the calculated model from above
predict(modFit, test_pca)
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The calculation of the accuracy is not possible since the classe variable is not included in
the testing data set. Since the in sample accuracy is about 97% and therefore the in sample
error is about 3% I expect a little higher out of sample error on the testing data.
