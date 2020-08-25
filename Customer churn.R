library(purrr)
library(dplyr)
library(ROSE)
library(DMwR)
library(e1071)
library(pROC)
library(ROCR)
library(varhandle)
library(caret)
library(tidyr)

#importing Dataset
churn_data <- read.csv("Churn_Modelling.csv" , row.names = 1)

########### Preprocessing and Transformation ###############

#dropping the id and surname columns from the dataset
churn_data <- churn_data[, -c(1,2)]

#Checking for missing values
sapply(churn_data, function(x) sum(is.na(x))) # No missing values found in the df

#Exploring the data
str(churn_data)
summary(churn_data)

# Plotting the data
churn_data %>%
  keep(negate(is.numeric)) %>%
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_bar()

churn_data %>%
  keep(is.numeric) %>%
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# we can convert 4 variables - exited,HasCrCard,isActiveMember and NumOfProducts into factors
churn_data <- churn_data %>% mutate(Exited = factor(Exited, labels =make.names(levels(factor(churn_data$Exited)))))
churn_data <- churn_data %>% mutate(HasCrCard = factor(HasCrCard, labels = make.names(levels(factor(churn_data$HasCrCard)))))
churn_data <- churn_data %>% mutate(IsActiveMember = factor(IsActiveMember, labels = make.names(levels(factor(churn_data$IsActiveMember)))))
churn_data <- churn_data %>% mutate(NumOfProducts = factor(NumOfProducts, labels = make.names(levels(factor(churn_data$NumOfProducts)))))

#Checking the dataset again
str(churn_data)
summary(churn_data)

#Creating test and training set
set.seed(1234)
t <- createDataPartition(churn_data$Exited, p=0.9, list = FALSE)
train_churn_orig <- churn_data[t,]
test_churn <- churn_data[-t,]

tr_control <- trainControl(method = "cv", number = 5,classProbs = TRUE, 
                           summaryFunction = twoClassSummary, verboseIter = TRUE)

############ Applying Models ######################

############ Applying logistic regression model on original train data #######
set.seed(1234)
glm1 <- train(Exited ~ .,data = train_churn_orig,
                    trControl = tr_control,
                    method = "glm",
                    family = "binomial")


# Confusion matrix of our orig. model on test data
pred_glm1 <- predict(glm1, test_churn)
CM_glm1_test <- confusionMatrix(pred_glm1,
                test_churn$Exited,
                positive = 'X1') # Very low specificity due to class imbalance
#Calculating auc of test
auc_glm1_test <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(glm1, test_churn)=='X0',0,1))

CM_glm1_test

# We can see our dependent variable 'Exited' is 4:1 imbalanced , so applying sampling techniques to tackle class imbalance

#######Using ROSE Sampling
set.seed(1111)
train_churn_rose <- ROSE(Exited ~ ., data  = train_churn_orig)$data                    
table(train_churn_rose$Exited) 

########### Applying logistic regression model on ROSE train data
set.seed(1234)
glm2_rose <- train(Exited ~ .,data = train_churn_rose,
                    trControl = tr_control,
                    method = "glm",
                    family = "binomial",preProcess = c('center', 'scale'))

# Confusion matrix of our ROSE model on test data
pred_glm2 <- predict(glm2_rose, test_churn)
CM_glm2_rose <- confusionMatrix(pred_glm2,
                           test_churn$Exited,
                           positive = 'X1')

auc_glm2_rose <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(glm2_rose, test_churn)=='X0',0,1))

CM_glm2_rose

##########Using SMOTE Sampling
set.seed(111)
train_churn_smote <- SMOTE(Exited ~ ., data  = train_churn_orig)                         
table(train_churn_smote$Exited)

######## Applying logistic regression model on SMOTE train data
set.seed(1234)
glm3_smote <- train(Exited ~ .,data = train_churn_smote,
                    trControl = tr_control,
                    method = "glm",
                    family = "binomial")

# Confusion matrix of our smote model on test data
pred_glm3<- predict(glm3_smote, test_churn)
CM_glm3_smote <- confusionMatrix(pred_glm3,
                           test_churn$Exited,
                           positive = 'X1')

auc_glm3_smote  <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(glm3_smote, test_churn)=='X0',0,1))
CM_glm3_smote



############## Applying Random Forest model on original train data ###############
set.seed(1234)
rf1 <- train(Exited ~ .,data = train_churn_orig, method = "ranger",metric = 'ROC',
                       trControl = tr_control)

# Confusion matrix of our orig. Random Forest model  on test data 
pred_rf1 <- predict(rf1, test_churn)
CM_rf1 <- confusionMatrix(pred_rf1,
                              test_churn$Exited,
                              positive = 'X1')

auc_rf1<- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(rf1, test_churn)=='X0',0,1))
CM_rf1




## Applying Random Forest model on ROSE train data
set.seed(1234)
rf2_rose <- train(Exited ~ .,data = train_churn_rose, method = "ranger",metric = 'ROC',
                       trControl = tr_control)


# Confusion matrix of our ROSE Random Forest model  on test data 
pred_rf2 <- predict(rf2_rose, test_churn)
CM_rf2_rose <- confusionMatrix(pred_rf2,
                              test_churn$Exited,
                              positive = 'X1')

auc_rf2_rose <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(rf2_rose, test_churn)=='X0',0,1))
CM_rf2_rose



## Applying Random Forest model on SMOTE train data
set.seed(1234)
rf3_smote <- train(Exited ~ .,data = train_churn_smote, method = "ranger",metric = 'ROC',
                       trControl = tr_control)


# Confusion matrix of our SMOTE Random Forest model  on test data 
pred_rf3 <- predict(rf3_smote, test_churn)
CM_rf3_smote <- confusionMatrix(pred_rf3,
                              test_churn$Exited,
                              positive = 'X1')

auc_rf3_smote <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(rf3_smote, test_churn)=='X0',0,1))

CM_rf3_smote


##### gradient boosted tree model on rose sampled data ####
set.seed(1234)
gbm2_rose <- train(Exited ~ .,data = train_churn_rose, method = "gbm",metric = 'ROC',
                  trControl = tr_control, preProcess = c('center', 'scale'))

# Confusion matrix of our ROSE gbm  model  on test data 
pred_gbm2 <- predict(gbm2_rose, test_churn)
CM_gbm2_rose <- confusionMatrix(pred_gbm2,
                               test_churn$Exited,
                               positive = 'X1')

auc_gbm2_rose <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(gbm2_rose, test_churn)=='X0',0,1))
CM_gbm2_rose


### neural net on rose sampled #######
set.seed(1234)
nn2_rose <- train(Exited ~ .,data = train_churn_rose, method = "nnet",metric = 'ROC',
                  trControl = tr_control,preProcess = c('center', 'scale'),
                  tuneGrid = expand.grid(.decay = c(0.1), .size = c(5)),maxit = 1000)

# Confusion matrix of our NN gbm  model  on test data 
pred_nn2 <- predict(nn2_rose, test_churn)
CM_nn2_rose <- confusionMatrix(pred_nn2,
                                test_churn$Exited,
                                positive = 'X1')

auc_nn2_rose <- auc(ifelse(test_churn$Exited == 'X0', 0,1), ifelse( predict(nn2_rose, test_churn)=='X0',0,1))
CM_nn2_rose



## Comparing 5 fold validatiion metrics of ORIG glm and RF models
model_list_orig <- list(glm1 = glm1, rf1 = rf1)
resamps_orig <- resamples(model_list_orig)
summary(resamps_orig)

## Comparing 5 fold validatiion metrics of rose glm and RF models
model_list_rose <- list(glm2_rose = glm2_rose, rf2_rose = rf2_rose, gbm2_rose = gbm2_rose, nn2_rose = nn2_rose)
resamps_rose <- resamples(model_list_rose)
summary(resamps_rose)

## Comparing 5 fold validatiion metrics of SMOTE glm and RF models
model_list_smote <- list(glm3_smote = glm3_smote, rf3_smote = rf3_smote)
resamps_smote <- resamples(model_list_smote)
summary(resamps_smote)

convertfactor <- function (x){
  return(ifelse(unfactor(x) == 'X0', 0,1))
}

preds_list <- list(convertfactor(pred_glm1), convertfactor(pred_glm2), convertfactor(pred_glm3),
                   convertfactor(pred_rf1), convertfactor(pred_rf2), convertfactor(pred_rf3),
                   convertfactor(pred_gbm2), convertfactor(pred_nn2))

m <- length(preds_list)
actuals_list <- rep(list(convertfactor(test_churn$Exited)), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("glm1", "glm2_rose","glm3_smote",
                  "rf1", "rf2_rose", "rf3_smote",
                  "gbm_rose","nn_rose"),fill = 1:m)




