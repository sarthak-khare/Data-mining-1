library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(psych)
library(caret)
library(rpart)
library(rpart.plot)
library(GGally)

house_df <- read.csv("kc_house_data.csv")

######## Data pre-processing and Transformation ####################

#Checking for missing values
sapply(house_df, function(x) sum(is.na(x))) # No missing values found in the df

#changing date to numeric
house_df$date<-(substr(house_df$date, 1, 8))
house_df$date<- ymd(house_df$date)
house_df$date<-as.numeric(as.Date(house_df$date, origin = "1900-01-01"))

#removing id columns as they do not impact our analyses
house_df_new <- house_df[,-c(1)]

## Checking if there is any relation of date with 
plot(house_df_new$date, house_df_new$price)

## Data exploration
str(house_df_new)
summary(house_df_new)

#checking the row for bedroom = 33 & 11 as it looks like an outlier/error
house_df_new[house_df_new$bedrooms == 33 | house_df_new$bedrooms == 11 , ]

# removing the row for bedroom = 33 and 11 as it definitely looks a mistake based on it's price , sqft_living etc.
house_df_new = house_df_new[(house_df_new$bedrooms != 11 & house_df_new$bedrooms != 33) , ]

# checking distribution of the variables
house_df_new %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# check scatterplots of all the variables against price
#house_df_new %>% 
#  gather(-price, key = "key", value = "value") %>% 
#  ggplot(aes(x=value, y= price)) +
#  geom_point() +
#  facet_wrap(~ key , scales = "free")

#Binning bathrooms by rounding values
house_df_new$bathrooms <- round(house_df_new$bathrooms)

# converting categorical variables to factors
house_df_new$bedrooms <- as.factor(house_df_new$bedrooms)
house_df_new$bathrooms <- as.factor(house_df_new$bathrooms)
house_df_new$condition <- as.factor(house_df_new$condition)
house_df_new$floors <- as.factor(house_df_new$floors)
house_df_new$waterfront <- as.factor(house_df_new$waterfront)
house_df_new$view <- as.factor(house_df_new$view)
house_df_new$grade <- as.factor(house_df_new$grade)

str(house_df_new)

# As dependent variable price is skewed , transforming by taking log
house_df_new$price <- log(house_df_new$price)

#Checking scatterplot of price vs dependent variables
plot(price ~ . , house_df_new)

# using pair.panels from psych package to check the correlations between numeric variables
ggcorr(house_df_new, name = "corr", label = TRUE, hjust = 1, label_size = 2.5, angle = -45, size = 3)

# From price plots and correlation matrix we can observe, date,zipcode, and long don't play any role , so dropping them
house_df_new <- house_df_new %>% select(-c(date, zipcode, long))

#year renovated can be converted to binary as most houses were not renovated
house_df_new$yr_renovated <- ifelse(house_df_new$yr_renovated == 0 , 0 , 1)
house_df_new$yr_renovated <- as.factor(house_df_new$yr_renovated)

# Scaling numeric variables using Z-score standardization
num_vars <- house_df_new %>% keep(is.numeric)
processed_vars <- preProcess(num_vars, method = c("center", "scale"))
house_df_new <- predict(processed_vars, house_df_new)

#creating a train - test split first
set.seed(11223)
t <- createDataPartition(house_df_new$price, p=0.8, list = FALSE)
train_house <- house_df_new[t,]
test_house <- house_df_new[-t,]





############### CREATING OUR MODEL #######################

##### MULTIPLE LINEAR MODEL #####

# Selecting numeric variables for our model that have correlation with price(>0.3) but not with each other
# we are left with sqft_living & lat
house_train_lm1 <-  train_house %>% select(c(price,sqft_living,lat,bedrooms,bathrooms,condition,floors,waterfront,view,grade))
house_test_lm1 <- test_house %>% select(c(price,sqft_living,lat,bedrooms,bathrooms,condition,floors,waterfront,view,grade))
# Creating our linear model using 10 fold cross-validation using caret package
lm1 <- train(price ~ . , house_train_lm1, method = "lm",
                trControl = trainControl(method = 'cv',number = 10, verboseIter = TRUE))
summary(lm1)

# Calculating test RMSE and R2
pred_price_lm1 <- predict(lm1, house_test_lm1)
R2_test_lm1 <- R2(pred_price_lm1 , test_house$price)
RMSE_test_lm1 <- RMSE(pred_price_lm1, test_house$price)


# As we are getting better fit on all variables, applying 10-fold cv
lm2 <- train(price ~ . , train_house, method = "lm",
                trControl = trainControl(method = 'cv',number = 10, verboseIter = TRUE))
summary(lm2)

# Calculating test RMSE and R2
pred_price_lm2 <- predict(lm2, test_house)
R2_test_lm2 <- R2(pred_price_lm2 , test_house$price)
RMSE_test_lm2 <- RMSE(pred_price_lm2, test_house$price)

# Creating a backward selection model
lm3 <- step(lm(price ~ . , train_house), direction = 'backward')
summary(lm3)

# Calculating test RMSE and R2
pred_price_lm3 <- predict(lm3, test_house)
R2_test_lm3 <- R2(pred_price_lm3 , test_house$price)
RMSE_test_lm3 <- RMSE(pred_price_lm3, test_house$price)


#creating a forward selection model
#lm4 <- train(price ~ . , house_train_lm1, method = "glmStepAIC",
#      trControl = trainControl(method = 'cv',number = 10, verboseIter = TRUE))
#summary(lm4)

##### DECISION TREE MODEL ##########

# For decision tree model we don't need our data to be scaled, so creating a new data frame
house_tree <- house_df[-1]
house_tree$price <- log(house_tree$price)

# creating training , validation and test datasets
set.seed(1)
assignment <- sample(1:3, size = nrow(house_tree), prob = c(0.7,0.1,0.2), replace = TRUE)
house_tree_train <- house_tree[assignment == 1, ]    # subset house_tree to training indices only
house_tree_valid <- house_tree[assignment == 2, ]  # subset house_tree to validation indices only
house_tree_test <- house_tree[assignment == 3, ]   # subset house_tree to test indices only

# Creating 1st model with default values of model parameters
tm1 <- rpart(formula = price ~ . , data = house_tree_train, method = "anova")
rpart.plot(x = tm1, yesno = 2, type = 0, extra = 0)

## Calculating train RMSE and R2


#creating prediction on test set
pred_tm1 <- predict(object = tm1,newdata = house_tree_test)
#Calculating R2 and RMSE
RMSE_tm1 <- RMSE(pred_tm1, house_tree_test$price)
R2_tm1 <- R2(pred_tm1, house_tree_test$price)

## Tuning the hyperparameters of the decision tree based on minsplit and max_depth
minsplit <- seq(10, 100, 10)
maxdepth <- seq(1, 10, 2)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(minsplit = minsplit, maxdepth = maxdepth)
num_models <- nrow(hyper_grid)

# Create an empty list to store models
house_tree_models <- list()
rmse_values <- c()

for (i in 1:num_models) {
  
  # Get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # Train a model and store in the list
  house_tree_models[[i]] <- rpart(formula = price ~ ., 
                             data = house_tree_train, 
                             method = "anova",
                             minsplit = minsplit,
                             maxdepth = maxdepth)
  pred <- predict(object = house_tree_models[[i]],newdata = house_tree_valid)
  
  rmse_values[i] <- RMSE(pred, house_tree_valid$price)
}

tm2 <- house_tree_models[[which.min(rmse_values)]]
tm2$control

# Compute test set RMSE on best_model
pred_tm2 <- predict(object = tm2,newdata = house_tree_test)
RMSE_tm2 <- RMSE(pred_tm2, house_tree_test$price )
R2_tm2 <- R2(pred_tm2, house_tree_test$price )

## Gradient boosted tree model
fitControl <- trainControl(method = "cv",
                           number = 5)
set.seed(825)
tm_gbm<- train(price ~ ., data = house_tree_train, 
                 method = "gbm", 
                 trControl = fitControl,
                 verbose = FALSE)

pred_gbm <- predict(object = tm_gbm,newdata = house_tree_test)
RMSE_gbm <- RMSE(pred_gbm, house_tree_test$price )
R2_gbm <- R2(pred_gbm, house_tree_test$price )

plot(tm1)