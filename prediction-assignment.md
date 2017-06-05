# Activity recognition prediction
Gennady Kalashnikov  

# Summary
This report describes creation of model that should be able to predict correctness of preformed exercise by data from accelerometers on the belt, forearm, arm, and dumbell.
Dataset is taken from "Human Activity Recognition project" (http://groupware.les.inf.puc-rio.br/har).

# Building model

## Loading data


```r
library(caret)
library(ElemStatLearn)
df_raw <- read.csv("pml-training.csv", na.strings=c("", "NA", "#DIV/0!"))
```

Original data provides a fairly large number of potential predictors 160, so at first the most valuable ones should be selected.

Simplify analysis by removing columns that are more than 95% NA.

```r
na_per_col <- sapply(df_raw, function(x){ sum(is.na(x)) })
full_cols <- names(na_per_col[na_per_col < 0.95 * nrow(df)])
df_full <- subset(df_raw, select = full_cols)
```

## Model selection

Original study grouped measurements in time windows and analysed them, but here objective is to build a model that will predict class for individual measurements.
So index, user_name, timestamp and window columns are removed.

```r
data_cols <- names(df_full)[!names(df_full) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "raw_timestamp_part_1", "cvtd_timestamp", "new_window", "num_window")]
df <- subset(df_full, select = data_cols)
```

There are still too many columns in data (53), so remove highly correlated ones.

```r
cor_cols <- findCorrelation(cor(df[,1:52]), cutoff = 0.7, names = TRUE)
df_c <- subset(df, select = names(df)[!names(df) %in% cor_cols])
```
Columns left: 31.

## Training and cross validation
Divide data into training, validation and test sets (in 60/20/20 proportion).


```r
set.seed(523)
inTrain = createDataPartition(df_c$classe, p = 0.6)[[1]]
training = df_c[ inTrain,]
inValidation = createDataPartition(df_c[ -inTrain,]$classe, p = 0.5)[[1]]
validation = df_c[ -inTrain,][ inValidation,]
testing = df_c[ -inTrain,][ -inValidation,]
```

Train one model using GBM (generalized boosted regression model) method and another one using random forest method.

```r
set.seed(634)
model_c_gbm <- train(classe ~ ., method="gbm", data = training, verbose = FALSE)
set.seed(745)
model_c_rf <- train(classe ~ ., method="rf", data = training)
```

These methods automatically tune some parameters and give accuracy evaluation.

```r
print(model_c_gbm)
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    30 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.6778157  0.5899151
##   1                  100      0.7308396  0.6585669
##   1                  150      0.7646508  0.7016096
##   2                   50      0.7788854  0.7193797
##   2                  100      0.8328191  0.7882356
##   2                  150      0.8616545  0.8248239
##   3                   50      0.8245467  0.7775774
##   3                  100      0.8752226  0.8419862
##   3                  150      0.9014860  0.8752632
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
print(model_c_rf)
```

```
## Random Forest 
## 
## 11776 samples
##    30 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9772911  0.9712528
##   16    0.9728767  0.9656674
##   30    0.9572873  0.9459318
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Anyway compare their accuracy on validation set.

```r
tncol = ncol(validation) - 1
gbm_acc_v <- sum(predict(model_c_gbm, validation[,1:tncol]) == validation$classe) / nrow(validation)
rf_acc_v <- sum(predict(model_c_rf, validation[,1:tncol]) == validation$classe) / nrow(validation)
```

Random forest accuracy (0.9873) on validation set is significantly better than accuracy of GBM method (0.9138).
High enough accuracy of random forest (together with the fact that classe is a discrete variable) make model ensembling unnecessary.
It's also worth noting that automatic accurary evaluations of boosting with trees and random forest methods were very close to accuracy on validation set.

Random forest model is selected as final model.

## Expected out of sample error

Calculate expected out of sample error as error of selected model on testing dataset.

```r
rf_acc_ev <- sum(predict(model_c_rf, testing[,1:tncol]) == testing$classe) / nrow(testing)
```
Accuracy - 0.9906

Error - 0.94%.
