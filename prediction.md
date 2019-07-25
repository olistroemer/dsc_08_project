---
title: "Exercise Manner Prediction Model"
author: "Oliver Stroemer"
date: "2019-07-24"
output:
        html_document:
                keep_md: true
---

## Synopsis

In this report we're try to find a prediction model, that predicts the manner in
which an weight lifting exercise was done. For this we're going to build
different models using different methods and choose the one performing best.
We'll use the [WLE data set](http://groupware.les.inf.puc-rio.br/har).

In the end we used a GBM with ten-fold cross-validation on our testing data and
got an accuracy of 100 %. The training set accuracy was 96 %. Of course we can
expect the out-of-sample error to be higher than both of these values.

## Loading libraries

To speed things up, we'll tell R to use multiple cores. Using the `doMC`
library, we'll register four cores for parallel processing.


```r
library(data.table)
library(caret)

library(doMC)
registerDoMC(4)
```


## Getting and Cleaning the Data

In this chapter, we're going to load the data from the internet and transform
them to our needs.


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              "pml-testing.csv")

training <- fread("pml-training.csv")
testing  <- fread("pml-testing.csv")
```

We can't use columns with missing data for our prediction models and we don't
have enough information about the data to impute the missing values. Therefore
we need to remove columns with missing data.


```r
training <- training[,colSums(is.na(training)) == 0, with=F]
```

Time dependent values are also irrelevant for our prediction since we can't
assume that there will be a relation between the time stamps in our data and out
of sample data. Also we are not interested in the user performing the exercise
for the same reason. We can cut our data down by the first seven columns.


```r
training <- training[,-c(1:7)]
```

The manner in which the exercise was done is saved in the `classe` column. We
need to convert the values to factors to be able to predict them.


```r
training$classe <- factor(training$classe)
```

## Exploratory Analysis

Let's have a look at whats left over from our original data.


```r
str(training) # Output redacted
```

There are still 52 predictors and our outcome of which we have almost 20000
observations! That's way too much for a pairs plot. Sad...

How many observations do we have per outcome category?


```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Category A has by far the most observations. Let's hope, that this unfair
distribution won't fall on our feet when we build our model.

## Building our Models

The easiest would have been to use a generalized linear model, but unfortunately
we can't use them, because we have more than two outcome categories. A decision
tree is great for data, we don't fully understand. We don't fully understand our
training data, we don't even know anything about it yet. So a tree seems like a
good idea. A tree has the benefit, that the model is relatively easy to
understand. It's possible to follow the path an observation takes by hand.

In order to make things reproducible, we have to set a seed. We'll use a numeric
representation of the date this report was written.


```r
set.seed(20190724)
```

### `rpart` model

Let's start with a single tree model. Its fast to compute and may do very well.
The data won't be pre-processed and we won't use any cross validation. But we'll
re-sample the data 25 times as that's the default option and we want to keep
things simple.


```r
m1 <- train(classe ~ ., method="rpart", data=training)
m1
```

```
## CART 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03567868  0.5103775  0.36287859
##   0.05998671  0.4229443  0.21926291
##   0.11515454  0.3234347  0.05936889
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03567868.
```

Ouch! That's bad... Only 0.5 accuracy. Its about two and a halve times better
than by chance ($\frac{1}{5} = 0.2$), but not good enough.

### Random Forest

Get the big cannons! We'll use the `rf` method, which means, we'll generate
many, many trees and combine them. Again we'll use the default options. This
might take a while, I'll get me a coffee.


```r
m2 <- train(classe ~ ., method="rf", data=training)
```

OK, by now I could have drunken more coffee than the supermarket has on stock.
Let's abort the computation and try something else.

### Gradient Boosting Machine

GBMs are very similar to Random Forests: both methods are based on decision
trees. Using the GBM the individual trees are independent from each other. Each
tree optimizes on the residuals of the predecessor. Again we'll use most of the
default options with no data pre-processing, but ten-fold cross-validation. The
ten-fold is a default if using the `cv` method in `train`.


```r
m3 <- train(classe ~ ., method="gbm", data=training,
            trControl=trainControl(method="cv"))
```

```r
m3
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17659, 17660, 17661, 17659, 17660, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7517588  0.6851596
##   1                  100      0.8200984  0.7722712
##   1                  150      0.8542955  0.8156133
##   2                   50      0.8565882  0.8182834
##   2                  100      0.9070415  0.8823786
##   2                  150      0.9329311  0.9151250
##   3                   50      0.8962893  0.8687112
##   3                  100      0.9443474  0.9295806
##   3                  150      0.9631526  0.9533819
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Gotcha! That's way better. The best model, the one with an interaction depth
(complexity of the tree) of 3 and 150 trees is about 96 % accurate. This is the
model, we'll use on the testing data set.

## Testing the Model

We have to input the results from the prediction manually in the Coursera Quiz
to validate them.


```r
predict(m3, newdata=testing) # Results hidden
```

Wow a **100 %** accuracy! I would've never expected that! Unfortunately we can
expect the out-of-sample error to be higher.

## Outlook

In a future model we can try to optimize the GBM a little bit more and don't
rely too much on the default options. It would be nice to see a readily built
random forest model on our testing data to compare it to the GBM; maybe it's
possible to reduce the amount of computation power needed tweaking the
parameters.
