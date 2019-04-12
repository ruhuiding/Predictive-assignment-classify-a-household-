# BAN404 Assignment 2

# clear the environment
rm(list = ls())

library(ElemStatLearn)
library(GoodmanKruskal)
library(MASS)
library(tree)
library(data.table)
library(plyr)
library(tidyverse) 
library(ggthemes)
library(e1071)
library(caret)
library(randomForest)

## a. Describe relevant features of the input and output variables with descriptive statistics. 
##    Create a variable high which is one for Income equal to $50,000 or more and zero otherwise.

?marketing
str(marketing)

# create a new variable high based on Income 
marketing$high <- ifelse(marketing$Income>=8, 1, 0)
# df: marketing dataset without na values
df <- na.omit(marketing[,-1])
# turn all the variables into factor
for(i in 1:ncol(df)){
  df[,i] <- as.factor(as.character(df[,i]))
  i<- i+1
}
str(df)

par(mfrow=c(3,5))
# plot the response with the predictors separately
for(i in 1:(ncol(df)-1)){
  barplot(table(df[,"high"], df[,i]), main=colnames(df)[i])
  i <- i+1
}

# turn off the plot settings if needed
graphics.off()

# Goodman and Kruskal???s tau measure for checking dependency
varset <- ls(marketing)
marketFrame <- as.data.frame(subset(marketing, select = varset)) 
GKmatrix <- GKtauDataframe(marketFrame)
plot(GKmatrix, corrColors="black")

## b. Use different methods to predict high.

### 1.  logistic regression

# with all variables
glm.fit1 = glm(high~., data=df, family=binomial)
summary(glm.fit1)

# Merge the last six levels into one level of variable Householdu18
# df2: dataset with merged values
df2 <- df
df2$Householdu18 <- revalue(df2$Householdu18,
                            c("4"="3","5"="3","6"="3","7"="3","8"="3","9"="3"))

# remove the predictors not helpful in predicing the response
glm.fit2 = glm(high~.-Lived-Ethnic-Language, data=df2, family=binomial)
#convert log odds to odds
Odds <- exp(coef(glm.fit2))
glm.odds <- cbind(round(summary(glm.fit2)$coefficient[,1], 3),
                  round(summary(glm.fit2)$coefficient[,4], 3),
                  round(Odds, 3))
colnames(glm.odds) <- c("Coefficient", "P-value", "Odds")
glm.odds

### 2. linear discriminant analysis

lda.fit = lda(high~., data=df2)
#Prior probabilities of groups
round(lda.fit$prior, 3)
cbind(t(round(lda.fit$means, 3)), #Group means
      round(lda.fit$scaling, 3)) #Coefficients of linear discriminants

### 3. classification trees with pruning

# fit a classification trees without pruning
tree.fit = tree(high~., data=df2)
plot(tree.fit)
text(tree.fit, pretty=0)

set.seed(3)
# perform cross validation to determine the optimal level of tree complexity
cv.fit = cv.tree(tree.fit,FUN=prune.misclass)
cv.fit
# prune the tree with the optimal tree complexity
prune.fit = prune.misclass(tree.fit, best=4)
plot(prune.fit)
text(prune.fit, pretty=0)

### 4. Compare the predictions with k-fold cross-validation

n <- nrow(df2)
K = 10
foldsize <- floor(n/K)
kk <- 1
#compute the fraction of observations for which the prediction was correct
# on test set of each fold
testCOR = matrix(NA, nrow=K, ncol=4)
colnames(testCOR) = c("glm1", "glm2", "lda", "tree")

testCOR0 = matrix(NA, nrow=K, ncol=4)
colnames(testCOR0) = c("glm1_0", "glm2_0", "lda_0", "tree_0")

testCOR1 = matrix(NA, nrow=K, ncol=4)
colnames(testCOR1) = c("glm1_1", "glm2_1", "lda_1", "tree_1")

for (i in 1:K){
  fold = kk:(kk+foldsize-1) # current fold
  kk <- kk+foldsize # next fold
  train <- df2[-fold, ] # current training set
  test <- df2[fold, ] # current test set
  
  # ****************** logistic regression 1 ++++++++++++++++++
  glm.fit1 = glm(high~., data=train, family=binomial)
  glm.probs = predict(glm.fit1, test, type="response")
  glm.pred = rep(0, nrow(test))
  glm.pred[glm.probs>.5] = 1
  con1 <- prop.table(table(glm.pred, test$high), margin=1)
  testCOR[i,"glm1"] <- round(mean(glm.pred==test$high), 5)
  testCOR0[i,"glm1_0"] <- round(con1[1,1], 5)
  testCOR1[i,"glm1_1"] <- round(con1[2,2], 5)
  
  # ****************** logistic regression 2 ++++++++++++++++++
  glm.fit2 = glm(high~.-Lived-Ethnic-Language, data=train, family=binomial)
  glm.probs2 = predict(glm.fit2, test, type="response")
  glm.pred2 = rep(0, nrow(test))
  glm.pred2[glm.probs2>.5] = 1
  con2 <- prop.table(table(glm.pred2, test$high), margin=1)
  testCOR[i,"glm2"] <- round(mean(glm.pred2==test$high), 5) 
  testCOR0[i,"glm2_0"] <- round(con2[1,1], 5)
  testCOR1[i,"glm2_1"] <- round(con2[2,2], 5)
  
  # ****************** linear discriminant analysis ++++++++++++++++++
  lda.fit = lda(high~., data=train)
  lda.pred = predict(lda.fit, test)
  lda.class = lda.pred$class
  con3 <- prop.table(table(lda.class, test$high), margin=1)
  testCOR[i,"lda"] <- round(mean(lda.class==test$high), 5)
  testCOR0[i,"lda_0"] <- round(con3[1,1], 5)
  testCOR1[i,"lda_1"] <- round(con3[2,2], 5)
  
  # ****************** classification trees with pruning ++++++++++++++++++
  tree.fit = tree(high~., train)
  prune.fit = prune.misclass(tree.fit, best=4)
  tree.pred = predict(tree.fit, test, type="class")
  con5 <- prop.table(table(tree.pred, test$high), margin=1)
  testCOR[i,"tree"] <- round(mean(tree.pred==test$high), 5)
  testCOR0[i,"tree_0"] <- round(con5[1,1], 5)
  testCOR1[i,"tree_1"] <- round(con5[2,2], 5)
}

testCOR
testCOR0
testCOR1
testCORS <- rbind(apply(testCOR,2,mean), apply(testCOR0,2,mean), apply(testCOR1,2,mean))
rownames(testCORS) <- c("Total accuracy rate", "Level 0 accuracy rate", "Level 1 accuracy rate")
testCORS

## c. Choose two additional prediction methods from the chapters 7-10 in James et al. (2015) 
##    Compare their predictions with the other methods.

### 1. Support Vector Machine (SVM)

## training data set and test set
set.seed(1111111)
n <- sample(nrow(df2), nrow(df2)/2, replace = FALSE)
train <- df2[n, ]
test <- df2[-n, ]

## cross validation in order to find the best model SVM
svm_radial <-  tune(svm, high~., data=train, kernel="radial") 
#, gamma = c(0.5, 1, 2, 3, 4),
#ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100)))

svm_radial$best.model ## select the best model from the cross validation

pred <- predict(svm_radial$best.model, newdata = test)

## compute the confusion matrix, accuracy of predictions on test data 80.37%
confusionMatrix(pred, test$high)

### 2. Bootstrap Aggregation (Bagging)

set.seed (1)
## bagging process
bag.high <- randomForest(high~., data=train, distribution="gaussian", mtry=13, importance=T)
summary(bag.high)

## Gini Index
bag.high$importance[, "MeanDecreaseGini"]

## plot the importance variable
par(mfrow=c(1, 1))
varImpPlot(bag.high, main="Importance Plots")

## predict high
yhat.bag = predict(bag.high, newdata=test, type="response")

## compute the confusion matrix, accuracy of prediction on test data 78.33 %
confusionMatrix(test$high, yhat.bag)

n <- nrow(df2)
K = 10
foldsize <- floor(n/K)
kk <- 1
## compute the fraction of observations for which the prediction was correct on test set of   each fold
testCOR = matrix(NA, nrow=K, ncol=2)
colnames(testCOR) = c("SVM", "bagging")
for (i in 1:K){
  fold = kk:(kk+foldsize-1) ## fold
  kk <- kk+foldsize ## next fold
  train <- df2[-fold, ] ## training set
  test <- df2[fold, ] ##  test set
  
  # ****************** Service Vector Machine ++++++++++++++++++
  svm_radial <- tune(svm, high~., data=train, kernel="radial") 
  pred <- predict(svm_radial$best.model, newdata=test)
  cM <- confusionMatrix(pred, test$high)
  testCOR[i,"SVM"] <- round(mean(cM[3]$overall["Accuracy"]), 5)
  
  # ****************** Bagging ++++++++++++++++++
  set.seed(1)
  bag.high <- randomForest(high~., data=train, distribution="gaussian", mtry=13, importance=T)
  yhat.bag = predict(bag.high, newdata=test, type="response")
  cM1 <- confusionMatrix(test$high, yhat.bag)
  testCOR[i,"bagging"] <- round(mean(cM1[3]$overall["Accuracy"]), 5)
}

testCOR
apply(testCOR, 2, mean)
