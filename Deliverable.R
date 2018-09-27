##### McKinsey Analytics Online Hackathon 2018 - Insurance #####


##### set working directory #####
setwd("C:/Users/hnguyen14/Box/McKinsey Analytics Online Competition 2018 - Insurance")

##### load packages #####
install.packages("foreign")
library(foreign)
install.packages("rpart")
library(rpart)
install.packages("partykit")
library(partykit)
install.packages("glmnet")
library(glmnet)
install.packages("e1071")
library(e1071)
install.packages("prodlim")
library(prodlim)
install.packages("caret")
library(caret)
install.packages("pROC")
library(pROC)
install.packages("randomForest")
library(randomForest)

# turn off warning messages
options(warn = -1)
# set seed for easy replication
set.seed(1)


##### read in the data file #####
train <- read.csv("train_ZoGVYWq.csv")
test <- read.csv("test_66516Ee.csv")


##### clean the data #####
# check the data type of each variable
sapply(train, class)
# convert variables to the correct data type
train$renewal <- as.factor(train$renewal)
# check the data type of each variable again
sapply(train, class)

summary(train)
summary(test)
# although the train set has a few outliers in the Income variable, it is important to keep these 
# outliers in the model because the test set has outliers too. The model needs to be trained on 
# extreme data points in order to have predictive power when used to predict the renewal rate in 
# the test set

# check for complete cases
nrow(train) # 79,853
sum(complete.cases(train)) # 76,855

# keep only complete rows in the train set for the model
df.analysis <- train[complete.cases(train), ]
summary(df.analysis)


##### candidate models #####
# classification models will be built based on a train set without any interaction variables, and a 
# train set with interaction variables. 
# candidate models include logistic regression, decision tree, and random forest


##### select variables for classification models without interaction variables #####
# for the train set without any interaction variables, a chisquare test will be used to test the 
# relationship between the outcome variable (renewal) and each independent variable.
chisq.test(df.analysis$renewal, df.analysis$perc_premium_paid_by_cash_credit) # significant
chisq.test(df.analysis$renewal, df.analysis$age_in_days) # significant
chisq.test(df.analysis$renewal, df.analysis$Income) # insignificant
chisq.test(df.analysis$renewal, df.analysis$Count_3.6_months_late) # significant
chisq.test(df.analysis$renewal, df.analysis$Count_6.12_months_late) # significant
chisq.test(df.analysis$renewal, df.analysis$Count_more_than_12_months_late) # significant
chisq.test(df.analysis$renewal, df.analysis$application_underwriting_score) # significant
chisq.test(df.analysis$renewal, df.analysis$no_of_premiums_paid) # significant
chisq.test(df.analysis$renewal, df.analysis$sourcing_channel) # significant
chisq.test(df.analysis$renewal, df.analysis$residence_area_type) # insignificant
chisq.test(df.analysis$renewal, df.analysis$premium) # significant
# although the relationships between renewal and income and residence_area_type are not 
# statistically significant at the 95% significance level, because the number of variables 
# in the train set is only 12, they will all be kept in the model building step.
# create a train set without interaction variables
drop <- "id"
df.no.interactions <- df.analysis[, !names(df.analysis) %in% drop]


##### build classification models with a 50-50 train-test split on df.no.interactions #####
# split df.no.interactions into train and test sets
test_set_50 <- sample(1:nrow(df.no.interactions), size = floor(nrow(df.no.interactions)/2))
test.50 <- df.no.interactions[test_set_50, ]
train.50 <- df.no.interactions[-test_set_50, ]

# check if the compositions of df.analysis (original train set with only complete cases), 
# test.50, and train.50 are the similar
table(df.analysis$renewal)/nrow(df.analysis)
table(train.50$renewal)/nrow(train.50)
table(test.50$renewal)/nrow(test.50)
# the compositions are the similar

# build classification models
model.logistic.50 <- glm(renewal ~ ., data = train.50, family = "binomial")
model.tree.50 <- rpart(renewal ~., data = train.50, method = "class")
model.forest.50 <- randomForest(renewal ~., data = train.50, ntree = 500, importance = TRUE)

# make predictions using classification models
pred.logistic.50 <- predict(model.logistic.50, newdata = test.50, type = "response")
pred.tree.50 <- predict(model.tree.50, newdata = test.50, type = "class")
pred.forest.50 <- predict (model.forest.50, newdata = test.50)

# build ROC curves
plot(roc(test.50$renewal, pred.logistic.50, direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for logistic regression model with 50-50 train-test split"))
plot(roc(test.50$renewal, as.numeric(pred.tree.50), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for decision tree model with 50-50 train-test split"))
plot(roc(test.50$renewal, as.numeric(pred.forest.50), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for random forest model with 50-50 train-test split"))

# calculate AUC
auc(test.50$renewal, pred.logistic.50) # 0.8266
auc(test.50$renewal, as.numeric(pred.tree.50)) # 0.5655
auc(test.50$renewal, as.numeric(pred.forest.50)) # 0.5612

# build confusion matrix
confusionMatrix(as.factor(ifelse(pred.logistic.50 >= 0.5, 1, 0)), test.50$renewal) # acc = 0.941
confusionMatrix(pred.tree.50, test.50$renewal) # 0.9399
confusionMatrix(pred.forest.50, test.50$renewal) # 0.9399

# if train-test split is 50-50, then logistic regression is the best model in terms of AUC 
# and accuracy


##### build classification models with a 75-25 train-test split on df.no.interactions #####
# split df.no.interactions into train and test sets
test_set_75 <- sample(1:nrow(df.no.interactions), size = floor(nrow(df.no.interactions)/4))
test.75 <- df.no.interactions[test_set_75, ]
train.75 <- df.no.interactions[-test_set_75, ]

# check if the compositions of df.analysis (original train set with only complete cases), 
# test.75, and train.75 are the similar
table(df.analysis$renewal)/nrow(df.analysis)
table(train.75$renewal)/nrow(train.75)
table(test.75$renewal)/nrow(test.75)
# the compositions are the similar

# build classification models
model.logistic.75 <- glm(renewal ~ ., data = train.75, family = "binomial")
model.tree.75 <- rpart(renewal ~., data = train.75, method = "class")
model.forest.75 <- randomForest(renewal ~., data = train.75, ntree = 500, importance = TRUE)

# make predictions using classification models
pred.logistic.75 <- predict(model.logistic.75, newdata = test.75, type = "response")
pred.tree.75 <- predict(model.tree.75, newdata = test.75, type = "class")
pred.forest.75 <- predict (model.forest.75, newdata = test.75)

# build ROC curves
plot(roc(test.75$renewal, pred.logistic.75, direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for logistic regression model with 75-25 train-test split"))
plot(roc(test.75$renewal, as.numeric(pred.tree.75), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for decision tree model with 75.25 train-test split"))
plot(roc(test.75$renewal, as.numeric(pred.forest.75), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for random forest model with 75.25 train-test split"))

# calculate AUC
auc(test.75$renewal, pred.logistic.75) # 0.8283
auc(test.75$renewal, as.numeric(pred.tree.75)) # 0.5641
auc(test.75$renewal, as.numeric(pred.forest.75)) # 0.5579

# build confusion matrix
confusionMatrix(as.factor(ifelse(pred.logistic.75 >= 0.5, 1, 0)), test.75$renewal) # acc = 0.94
confusionMatrix(pred.tree.75, test.75$renewal) # 0.9393
confusionMatrix(pred.forest.75, test.75$renewal) # 0.9386

# if train-test split is 75-25, then logistic regression is the best model in terms of AUC 
# and accuracy


##### build classification models with a 80-20 train-test split on df.no.interactions #####
# split df.no.interactions into train and test sets
test_set_80 <- sample(1:nrow(df.no.interactions), size = floor(nrow(df.no.interactions)/5))
test.80 <- df.no.interactions[test_set_80, ]
train.80 <- df.no.interactions[-test_set_80, ]

# check if the compositions of df.analysis (original train set with only complete cases), test.80, and
# train.80 are the similar
table(df.analysis$renewal)/nrow(df.analysis)
table(train.80$renewal)/nrow(train.80)
table(test.80$renewal)/nrow(test.80)
# the compositions are the similar

# build classification models
model.logistic.80 <- glm(renewal ~ ., data = train.80, family = "binomial")
model.tree.80 <- rpart(renewal ~., data = train.80, method = "class")
model.forest.80 <- randomForest(renewal ~., data = train.80, ntree = 500, importance = TRUE)

# make predictions using classification models
pred.logistic.80 <- predict(model.logistic.80, newdata = test.80, type = "response")
pred.tree.80 <- predict(model.tree.80, newdata = test.80, type = "class")
pred.forest.80 <- predict (model.forest.80, newdata = test.80)

# build ROC curves
plot(roc(test.80$renewal, pred.logistic.80, direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for logistic regression model with 80-20 train-test split"))
plot(roc(test.80$renewal, as.numeric(pred.tree.80), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for decision tree model with 80-20 train-test split"))
plot(roc(test.80$renewal, as.numeric(pred.forest.80), direction = "<", col = "blue", lwd = 3, 
         main = "ROC curve for random forest model with 80-2- train-test split"))

# calculate AUC
auc(test.80$renewal, pred.logistic.80) # 0.8271
auc(test.80$renewal, as.numeric(pred.tree.80)) # 0.5598
auc(test.80$renewal, as.numeric(pred.forest.80)) # 0.5559

# build confusion matrix
confusionMatrix(as.factor(ifelse(pred.logistic.80 >= 0.5, 1, 0)), test.80$renewal) # acc = 0.9396
confusionMatrix(pred.tree.80, test.80$renewal) # 0.9396
confusionMatrix(pred.forest.80, test.80$renewal) # 0.9397

# if train-test split is 80-20, then logistic regression is the best model in terms of AUC while
# random forest is the best model in terms of accuracy


##### final chosen model based on a train set with no interaction variables #####
# based on the AUC ROC scores, the logistic regression model performed the best among different
# train-test splits in the model selection step
# Therefore, the logistic regression model will be used on all of the train set in 
# df.no.interaction to predict the renewal rate in the orignal test set

# build the model
model.logistic.all <- glm(renewal ~., data = df.no.interactions, family = "binomial")

# make predictions
pred.logistic.all <- predict(model.logistic.all, newdata = test, type = "response")
test$renewal_probability <- predict(model.logistic.all, newdata = test, type = "response")
test$renewal_predicted <- as.factor(ifelse(test$renewal_probability >= 0.5, 1, 0))

summary(test)
head(test)

# based on the logistic regression model, we were able to predict the likelihood of customers
# renewing their insurance policies, except for the customers with incomplete data.