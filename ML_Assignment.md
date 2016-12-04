Synopsis
--------

We will have two datasets of the usage of personal devices for example Jowbone, fitbit and NiekFuelBand-- one for training and another one is for testing.

Please ignore the warning: Warning in FUN(X\[\[i\]\], ...) proceduce by building prediction for the Naive Bayes model. I really want to know why I have got these warnings. I appreciate If you can give me a feedback.

Load data
---------

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.3.2

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(RCurl)
```

    ## Loading required package: bitops

``` r
library(doParallel)
```

    ## Warning: package 'doParallel' was built under R version 3.3.2

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 3.3.2

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 3.3.2

    ## Loading required package: parallel

``` r
file <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
training <- read.csv(text=file, header = T, sep=",", na.strings=c("#DIV/0!", "", "NA"))

file <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

testing <- read.csv(text=file, header = T, sep = ",", na.strings=c("#DIV/0!", "", "NA"),stringsAsFactors = F)
```

Peaking into the dataset
------------------------

I have checked the training data set by using summary method and I have found a lot of missing values. Also a weird notation like \#DIV/0! which I have no idea what it is. For \#DIV/0!, I have decided to convert it to NA value by using na.strings when reading csv file for both training and test dataset.

Cleaning Strategy
-----------------

``` r
ratioMissing <- function(x){sum(is.na(x))/length(x)*100}
ratioTraining <- apply(training,2,ratioMissing)
ratioTesting <- apply(testing,2,ratioMissing)
```

I have created a function which basically can show us the ratio of missing values in each variable and the result doesn't look good-- I meant, there are too many missing values, approximately 98 - 100%.

I don't think we can impute these columns as the missing ratios are really high. We have two options, either make them zero or exclude them. I have decided to exclude them by comparing with the ratio I have got and we will do the same for the testing set.

``` r
newTraining <- training[, ratioTraining < 90]
newTesting <- testing[, ratioTesting < 90]
```

I did try using X, username and timestamp variables to train models before but it didn't work and turned out my final model was too biased. So I have to remove those variables.

``` r
newTraining <- subset(newTraining, select=-c(1:7))
newTesting <- subset(newTesting, select=-c(1:7))
```

Then I separate the training set into training set and validation set.

``` r
set.seed(7812)
inTrain <- createDataPartition(y=newTraining$classe, p = 0.7,list=FALSE)
finalTraining <- newTraining[inTrain,]
validation <- newTraining[-inTrain,]
```

Parallel setting
----------------

I am writing R code in Windows and my laptop is quite old. Instead a default CPU processing I will use parallel technique to boost up the speed by using doParallel package.

``` r
registerDoParallel(makeCluster(4))
```

Choosing Model
--------------

From the list [HERE](https://topepo.github.io/caret/available-models.html), also our outcome is 5 classesr, and among the famous models in the lecture, I would like to see how good amid NaiveBayes(nb), Recursive Paritioning and Regression Tree(rpart) and Random Forest(rf) are. Actually I wanted to include gbm but I have tried several times and my computer crashed while running gbm. We first compare the models using default resampling which is boottrap without preprocessing.

``` r
set.seed(1235)
model1 <- train(classe ~ ., data=finalTraining, method="nb")
```

    ## Loading required package: klaR

    ## Warning: package 'klaR' was built under R version 3.3.2

    ## Loading required package: MASS

``` r
set.seed(1235)
model2 <- train(classe ~ ., data=finalTraining, method="rpart")
```

    ## Loading required package: rpart

``` r
set.seed(1235)
model3 <- train(classe ~ ., data=finalTraining, method="rf")
```

    ## Loading required package: randomForest

    ## Warning: package 'randomForest' was built under R version 3.3.2

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

The accuracy for each model:

### nb

``` r
confusionMatrix(predict(model1, newdata=finalTraining), finalTraining$classe)
```

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 1

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 2

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 3

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 4

    ## Warning in FUN(X[[i]], ...): Numerical 0 probability for all classes with
    ## observation 5

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3485  527  522  428  146
    ##          B   77 1764  157    4  210
    ##          C  115  207 1613  296   92
    ##          D  209  135   97 1426   92
    ##          E   20   25    7   98 1985
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7478          
    ##                  95% CI : (0.7405, 0.7551)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6774          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8922   0.6637   0.6732   0.6332   0.7861
    ## Specificity            0.8349   0.9596   0.9374   0.9536   0.9866
    ## Pos Pred Value         0.6823   0.7975   0.6944   0.7279   0.9297
    ## Neg Pred Value         0.9512   0.9224   0.9314   0.9299   0.9535
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2537   0.1284   0.1174   0.1038   0.1445
    ## Detection Prevalence   0.3718   0.1610   0.1691   0.1426   0.1554
    ## Balanced Accuracy      0.8636   0.8116   0.8053   0.7934   0.8864

### rpart

``` r
confusionMatrix(predict(model2, newdata=finalTraining), finalTraining$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3537 1067 1087  984  347
    ##          B   79  942   95  428  377
    ##          C  282  649 1214  840  663
    ##          D    0    0    0    0    0
    ##          E    8    0    0    0 1138
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4973          
    ##                  95% CI : (0.4889, 0.5057)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3436          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9055  0.35440  0.50668   0.0000  0.45069
    ## Specificity            0.6455  0.91163  0.78538   1.0000  0.99929
    ## Pos Pred Value         0.5037  0.49037  0.33279      NaN  0.99302
    ## Neg Pred Value         0.9450  0.85477  0.88284   0.8361  0.88984
    ## Prevalence             0.2843  0.19349  0.17442   0.1639  0.18381
    ## Detection Rate         0.2575  0.06857  0.08837   0.0000  0.08284
    ## Detection Prevalence   0.5112  0.13984  0.26556   0.0000  0.08342
    ## Balanced Accuracy      0.7755  0.63302  0.64603   0.5000  0.72499

### randomForest

``` r
confusionMatrix(predict(model3, newdata=finalTraining), finalTraining$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3906    0    0    0    0
    ##          B    0 2658    0    0    0
    ##          C    0    0 2396    0    0
    ##          D    0    0    0 2252    0
    ##          E    0    0    0    0 2525
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

From this result, I will choose the Random Forest to me my final model as it has smallest in-sample error(1 - accuracy), to train the training data with repeated 10 k-folds cross-validation.

``` r
# Setting the resampling technique in the traing control.
train_Con <- trainControl(method="repeatedcv", number=10, repeats=5)

# Train the data with the training control using rf method.
set.seed(5632411)
best.mod <-  train(classe ~ ., data=finalTraining, method="rf", trControl=train_Con)
```

Now we compare to the validation set to check how accurate the model is.

``` r
confusionMatrix(predict(best.mod, newdata=validation), validation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    7    0    0    0
    ##          B    0 1131   17    0    0
    ##          C    0    1 1009   14    1
    ##          D    0    0    0  950    3
    ##          E    0    0    0    0 1078
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9927          
    ##                  95% CI : (0.9902, 0.9947)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9908          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9930   0.9834   0.9855   0.9963
    ## Specificity            0.9983   0.9964   0.9967   0.9994   1.0000
    ## Pos Pred Value         0.9958   0.9852   0.9844   0.9969   1.0000
    ## Neg Pred Value         1.0000   0.9983   0.9965   0.9972   0.9992
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1922   0.1715   0.1614   0.1832
    ## Detection Prevalence   0.2856   0.1951   0.1742   0.1619   0.1832
    ## Balanced Accuracy      0.9992   0.9947   0.9901   0.9924   0.9982

The out error is around 1-0.9998 = 0.1% which is quite good. Lets see what variables that affect this accuracy.

``` r
varImp(best.mod, scale = F)
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                   Overall
    ## roll_belt           498.0
    ## yaw_belt            435.5
    ## magnet_dumbbell_z   378.9
    ## magnet_dumbbell_y   350.3
    ## pitch_belt          345.8
    ## pitch_forearm       338.7
    ## magnet_dumbbell_x   308.7
    ## roll_forearm        303.2
    ## accel_dumbbell_y    274.8
    ## magnet_belt_z       268.7
    ## roll_dumbbell       267.2
    ## magnet_belt_y       264.4
    ## accel_belt_z        262.7
    ## accel_dumbbell_z    253.8
    ## roll_arm            232.8
    ## accel_forearm_x     229.2
    ## accel_arm_x         213.3
    ## yaw_dumbbell        210.2
    ## accel_dumbbell_x    209.4
    ## gyros_belt_z        204.4

``` r
varImpPlot(best.mod$finalModel, type=2)
```

![](ML_Assignment_files/figure-markdown_github/unnamed-chunk-13-1.png)

Now lets make a prediction with the 20 observation test set.

``` r
pre.test <- predict(best.mod, newdata=newTesting)
pre.test
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Summary
-------

After trying three different models, rpart is the fastest model that I have tried but for the accuracy, Random Forest is at the first place, then nb and rpart respectively.

For the final model, rf, The in-sample error is optimistically less than out-sample error.

I didn't use any preprocess techniques because I was building a classification model where transforming data is not required.
