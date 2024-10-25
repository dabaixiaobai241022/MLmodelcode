rm(list=ls())

library(lme4)
library(lmerTest)
library(emmeans)
library(msm)
library(vctrs)
library(ipw)
library(tidyverse)
library(survival)
library(xgboost)
library(caret)
library(MASS)
library(Matrix)
library(foreach)
library(doParallel)
library(data.table)
library(tidymodels)

memory.limit(500000000)
gc()

# Define working directory  
setwd("xx")
# Read data
lipidsdata <- fread(
  "xx.csv",
  stringsAsFactors=FALSE,
  encoding = "UTF-8"
)

lipidsdata$outcome <-  as.numeric(
  as.character(lipidsdata$outcome))

# Parameter tuning grid
grid <- expand.grid(
  nrounds = 100 ,
  max_depth = c(3,4,5,6,7,8),
  eta = c(0.01,0.02,0.03,0.06,0.1,0.2,0.3),
  gamma = c(0,0.01,0.05,0.1,0.2,0.5,1,2),
  colsample_bytree =  c(0.5,0.6,0.7,0.8,0.9,1),
  min_child_weight = c(1:6),
  subsample =  c(0.5,0.6,0.7,0.8,0.9,1)
)

# Split dataset, 80% for training set, 20% for testing set
set.seed(123)
lipidsdata_split <- initial_split(lipidsdata, prop = 0.80)
lipidsdata_train <- training(lipidsdata_split)
lipidsdata_test <- testing(lipidsdata_split)

## Training set

lipidsdata_train_td <- data.matrix(lipidsdata_train[, x1:x2]) %>%
  Matrix(., sparse = T) %>%
  list(data = ., y = lipidsdata_train$outcome)

lipidsdata_train_matrix <- xgb.DMatrix(
  data = lipidsdata_train_td$data,
  label = lipidsdata_train_td$y,
  weight = lipidsdata_train$Weight1
)

## Testing set

lipidsdata_test_td <- data.matrix(lipidsdata_test[, x1:x2]) %>%
  Matrix(., sparse = T) %>%
  list(data = ., y = as.factor(lipidsdata_test$outcome))

lipidsdata_test_matrix <- xgb.DMatrix(
  data = lipidsdata_test_td$data,
  label = lipidsdata_test_td$y,
  weight = lipidsdata_test$Weight1
)

# Looping  ----------------------------------------------------------------------

options(warn = -1)

cl <- makePSOCKcluster(10) 
registerDoParallel(cl)

results_all <- map(
  1:1000,
  function(i) {
    
    print(str_glue("Start cross-validation with seed number {i}--------------------------------"))
    
    # Cross-validation
    
    set.seed(i)
    random_search <- train(
      outcome ~Covariates,
      data = lipidsdata_train,
      method = "xgbTree",
      tuneGrid = grid,
      metric = "RMSE",       # Select evaluation metrics
      trControl = trainControl(
        method = "cv",       # Cross-validation
        number = 10,         # Number of folds
        allowParallel = T    # Allow parallel processing
      )
    )
    best_params <- random_search$bestTune      # Final model parameters
    tune_results <- random_search$results %>%  # All evaluation metrics
      mutate(seed = i)
    
    # Fit final model
    
    mod <- xgb.train(
      data = lipidsdata_train_matrix,
      objective = "binary:logistic",
      nrounds = best_params$nrounds,
      max_depth = best_params$max_depth,
      eta = best_params$eta,
      gamma = best_params$gamma,
      colsample_bytree = best_params$colsample_bytree,
      min_child_weight = best_params$min_child_weight,
      subsample = best_params$subsample
    )
    
    # Predictions on training set
    
    predicted_train_f <- predict(
      mod, newdata = lipidsdata_train_matrix, weights = lipidsdata$Weight1
    ) %>%
      as_tibble() %>%
      `colnames<-`(str_c("prediction_", i))
    
    # Predictions on testing set
    predicted_test_f <- predict(
      mod, newdata = lipidsdata_test_matrix, weights = lipidsdata$Weight1
    ) %>%
      as_tibble() %>%
      `colnames<-`(str_c("prediction_", i))
    
    # Output model predictions and evaluation metrics 
    
    results_f <- list(
      "Predictions on training set" = predicted_train_f,
      "Predictions on testing set" = predicted_test_f,
      "Evaluation metrics" = tune_results
    )
    
    print(str_glue("Complete cross-validation with seed number {i} -----------------------------------"))
    
    return(results_f)
  })

stopCluster(cl)

options(warn = 1) # open warning


predicted_train_results <- lipidsdata_train %>%
  bind_cols(
    results_all %>%
      map(~ .x[["Predictions on training set"]]) %>%
      reduce(bind_cols))

predicted_test_results <- lipidsdata_test %>%
  bind_cols(
    results_all %>%
      map(~ .x[["Predictions on testing set"]]) %>%
      reduce(bind_cols))


metric_all <- results_all %>%
  map(~ .x[["Evaluation metrics"]]) %>%
  reduce(bind_rows)

# Save data 
write.csv(predicted_train_results, "xx.csv" )
write.csv(metric_all,"xx.csv" )
write.csv( predicted_test_results,"xx.csv" )

