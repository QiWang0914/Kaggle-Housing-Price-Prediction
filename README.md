# Kaggle-Housing-Price-Prediction

train <- read.csv("C:/Users/Mr.Q/Documents/train.csv",stringsAsFactors = FALSE)
trainCont <- train[sapply(train,is.numeric)]#put numerical features into matrix
trainChar <- train[sapply(train,is.character)]
new.trainCont <- subset(trainCont, select = -c(Id,MoSold,MSSubClass,OverallQual,OverallCond))#remove non-continuous coloumns from trainCont
correlations <- cor(new.trainCont, use = "pairwise.complete.obs")#calculate continuous features correlations 
trainCont_correlations <- correlations
ifelse(abs(trainCont_correlations < 0.5), NA, trainCont_correlations )

ABS_trainCont_correlations <- abs(trainCont_correlations)#calculate absolute value of correlations
Select1 <- trainCont_correlations[,"SalePrice"]
ABS_trainCont_correlations_onPrice <- ABS_trainCont_correlations[,"SalePrice"]

sort(rank(ABS_trainCont_correlations_onPrice))[1:5]

#Linear Regression

setwd("C:/Users/Mr.Q/Documents")
train <- read.csv("train.csv", stringsAsFactors = FALSE)
train$LogPrice <- log(train$SalePrice)
##feature engineering: house total area
train$TotalArea <- train$GrLivArea + train$TotalBsmtSF + train$GarageArea+
                   train$LotArea + train$MasVnrArea + train$OpenPorchSF+
                   train$PoolArea + train$ScreenPorch + train$WoodDeckSF+
                   train$X3SsnPorch + train$EnclosedPorch
##feature engineering: numbers of bathroom
train$TotalBath <- train$BsmtFullBath + 0.5*train$BsmtHalfBath +
                   train$FullBath + 0.5*train$HalfBath
##interval of remodel
train$IntervalRemodel <- train$YearRemodAdd - train$YearBuilt
dim(train)

na_value <- sort(sapply(train, function(x) { sum(is.na(x)) }), decreasing = FALSE)
na_value

Fin_train <- names(which(na_value < dim(train)[1] * 0.05))
length(Fin_train) 
train <- train[, c(Fin_train)]

##dealing with missing value
train[which(is.na(train$BsmtExposure)),c('BsmtExposure','BsmtFinType1','BsmtFinSF1',
                                         'BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
                                         'BsmtQual','BsmtCond')]
                                         
train$BsmtExposure[which(is.na(train$BsmtExposure))] <- 'Unfinished'
train$BsmtFinType1[which(is.na(train$BsmtFinType1))] <- 'Unfinished'
train$BsmtFinType2[which(is.na(train$BsmtFinType2))] <- 'Unfinished'
train$BsmtQual[which(is.na(train$BsmtQual))] <- 'Unfinished'
train$BsmtCond[which(is.na(train$BsmtCond))] <- 'Unfinished'

train$MasVnrArea[which(is.na(train$MasVnrArea))] <- mean(train$MasVnrArea, na.rm = TRUE)

library(mice)
impute.train <- mice(train, m = 5, printFlag = FALSE)
## note the categorical (character) variable needs to be factor.
train$MasVnrType <- as.factor(train$MasVnrType)
train$Electrical <- as.factor(train$Electrical)
impute.train <- mice(train, m = 5, method='pmm', printFlag = FALSE)
train_update <- complete(impute.train)
table(train$TotalArea)
#saved to no missing value file 'train_update.csv'
write.csv(train_update, file = "train_update.csv", row.names = FALSE)

library(Matrix)
library(foreach)
library(glmnet)

cate_feature <- c("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
                  "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                  "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                  "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                  "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
                  "Fence", "MiscFeature", "MoSold", "SaleType", "SaleCondition")

Feature_name <- names(train)
sort(Feature_name)

cate_feature_update <- c("MSSubClass", "MSZoning", "Street", "LotShape", "LandContour",
                         "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                         "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                         "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                         "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                         "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", 
                         "PavedDrive", 
                         "MoSold", "SaleType", "SaleCondition")
train[cate_feature_update] <- lapply(train[cate_feature_update], as.factor)
ind <- model.matrix(~., subset(train, select = -c(Id, SalePrice,LogPrice)))
dep <- train$LogPrice
table(train$MSZoning)
head(ind)

set.seed(100)
train.ind <- sample(1:dim(ind)[1], dim(ind)[1] * 0.7)
x.train <- ind[train.ind,]
x.test <- ind[-train.ind,]
y.train <- dep[train.ind]
y.test <- dep[-train.ind]

fit.lasso <- glmnet(x.train, y.train, alpha = 1)
fit.ridge <- glmnet(x.train, y.train, alpha = 0)
fit.elnet <- glmnet(x.train, y.train, alpha = 0.5)

fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 1, family = "gaussian")
fit.ridge.cv <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0, family = "gaussian")
fit.elnet.cv <- cv.glmnet(x.train, y.train, type.measure = "mse", alpha = 0.5, family = "gaussian")

for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train, y.train, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

par(mfrow = c(3,2))
plot(fit.lasso, xvar = "lambda")
plot(fit10, main = "LASSO")

plot(fit.ridge, xvar = "lambda")
plot(fit10, main = "Ridge")

plot(fit.elnet, xvar = "lambda")
plot(fit10, main = "Elastic Net")

for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x.train, y.train, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

y_pred0 <- predict(fit0, s = fit0$lambda.1se, newx = x.test)
y_pred0

y_pred1 <- predict(fit1, s=fit1$lambda.1se, newx=x.test)
y_pred2 <- predict(fit2, s=fit2$lambda.1se, newx=x.test)
y_pred3 <- predict(fit3, s=fit3$lambda.1se, newx=x.test)
y_pred4 <- predict(fit4, s=fit4$lambda.1se, newx=x.test)
y_pred5 <- predict(fit5, s=fit5$lambda.1se, newx=x.test)
y_pred6 <- predict(fit6, s=fit6$lambda.1se, newx=x.test)
y_pred7 <- predict(fit7, s=fit7$lambda.1se, newx=x.test)
y_pred8 <- predict(fit8, s=fit8$lambda.1se, newx=x.test)
y_pred9 <- predict(fit9, s=fit9$lambda.1se, newx=x.test)
y_pred10 <- predict(fit10, s=fit10$lambda.1se, newx=x.test)

coef(fit0, s = "lambda.1se")

mod0 <- lm(LogPrice ~., data = train)
mod0
#--------------------------------------------------------------------------------------------------------------------------------
## tree based model
train <- read.csv("train.csv", stringsAsFactors = FALSE)
train <- subset(train, select = -c(Id))#delete id
test <-  read.csv("test.csv", stringsAsFactors = FALSE)
test$SalePrice<-0 #add a saleprice column to make it the same feature column with train 
test<- test[,-1] #delete id
head(test)
Totset <- rbind(train, test)
dim(Totset)
na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value

Totset$PoolQC[is.na(Totset$PoolQC)] <- 'No'
Totset$MiscFeature[is.na(Totset$MiscFeature)] <- 'No'
Totset$Alley[is.na(Totset$Alley)] <- 'No'
Totset$Fence[is.na(Totset$Fence)] <- 'No'
Totset$FireplaceQu[is.na(Totset$FireplaceQu)] <- 'No'
Totset$GarageFinish[is.na(Totset$GarageFinish)] <- 'No'
Totset$GarageQual[is.na(Totset$GarageQual)] <- 'No'
Totset$GarageCond[is.na(Totset$GarageCond)] <- 'No'
Totset$GarageType[is.na(Totset$GarageType)] <- 'No'
Totset$BsmtCond[is.na(Totset$BsmtCond)] <- 'No'
Totset$BsmtExposure[is.na(Totset$BsmtExposure)] <- 'No'
Totset$BsmtQual[is.na(Totset$BsmtQual)] <- 'No'
Totset$BsmtFinType2[is.na(Totset$BsmtFinType2)] <- 'No'
Totset$BsmtFinType1[is.na(Totset$BsmtFinType1)] <- 'No'
na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value

Fin_Totset <- names(which(na_value < dim(Totset)[1] * 0.05))
Totset <- Totset[, Fin_Totset]
dim(Totset)

na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value

Totset$BsmtFullBath[is.na(Totset$BsmtFullBath)] <- 0
Totset$BsmtHalfBath[is.na(Totset$BsmtHalfBath)] <- 0
Totset$BsmtFinSF1[is.na(Totset$BsmtFinSF1)] <- 0
Totset$BsmtFinSF2[is.na(Totset$BsmtFinSF2)] <- 0
Totset$BsmtUnfSF[is.na(Totset$BsmtUnfSF)] <- 0
Totset$TotalBsmtSF[is.na(Totset$TotalBsmtSF)] <- 0
Totset$GarageCars[is.na(Totset$GarageCars)] <- 0
Totset$GarageArea[is.na(Totset$GarageArea)] <- 0
na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value

Totset[which(is.na(Totset$Electrical)),]
Totset$MasVnrArea[which(is.na(Totset$MasVnrArea))] <- mean(Totset$MasVnrArea, na.rm = T)
na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value
bdcol <- "Electrical"
Totset <- Totset[, !names(Totset) %in% bdcol, drop = FALSE]
bdcol <- "Utilities"
Totset <- Totset[, !names(Totset) %in% bdcol, drop = FALSE]
dim(Totset)
Totset$MasVnrType[is.na(Totset$MasVnrType)] <- 0
Totset$MSZoning[is.na(Totset$MSZoning)] <- 0
Totset$SaleType[is.na(Totset$SaleType)] <- 0
Totset$Functional[is.na(Totset$Functional)] <- 0
Totset$Exterior1st[is.na(Totset$Exterior1st)] <- 0
Totset$Exterior2nd[is.na(Totset$Exterior2nd)] <- 0
Totset$KitchenQual[is.na(Totset$KitchenQual)] <- 0
na_value <- sort(sapply(Totset, function(x) { sum(is.na(x)) }), decreasing = T)
na_value
sort(sapply(Totset, function(x) {sum(is.na(x))}), decreasing = FALSE)
Totset$LogPrice <- log(Totset$SalePrice)
##feature engineering: house total area
Totset$TotalArea <- Totset$GrLivArea + Totset$TotalBsmtSF + Totset$GarageArea+
  Totset$LotArea + Totset$MasVnrArea + Totset$OpenPorchSF+
  Totset$PoolArea + Totset$ScreenPorch + Totset$WoodDeckSF+
  Totset$X3SsnPorch + Totset$EnclosedPorch
##feature engineering: numbers of bathroom
Totset$TotalBath <- Totset$BsmtFullBath + 0.5*Totset$BsmtHalfBath +
  Totset$FullBath + 0.5*Totset$HalfBath
dim(Totset)

#Set factor for all characters
catg_feature <- c("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
                  "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
                  "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",
                  "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
                  "HeatingQC", "CentralAir", "KitchenQual", "Functional", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
                  "Fence", "MiscFeature", "MoSold", "SaleType", "SaleCondition")
Feature_name <- names(Totset)
setdiff(catg_feature, Feature_name)
##Spliting Totset into train.data and test.data with “caret” package and fitting with randomForest
library(caret)
library(lattice)
library(ggplot2)
index <- createDataPartition(Totset$LogPrice, time=1, p=0.7, list=F)
train.data <- Totset[index,]
test.data <- Totset[-index,]
dim(train.data)
dim(test.data)
for(i in 1:(dim(train.data)[2])){
  if(is.character(train.data[, i]) & 
     length(which(!unique(test.data[, i]) %in% unique(train.data[, i]))) > 0) {
    print(paste("this column: ", colnames(train.data)[i], "has new levels in test"))
  } 
}

##fitting with random forest
library(randomForest)
set.seed(100)
for(i in 1:dim(Totset)[2]) {
  if(is.character(Totset[,i])) {
    Totset[,i] <- as.factor(Totset[,i])
  }
}
index <- createDataPartition(Totset$LogPrice, time=1, p=0.7, list=F)
train.data <- Totset[index,]
test.data <- Totset[-index,]
str(train.data)

rf.formula <- "log(SalePrice) ~ .-SalePrice "
rf <- randomForest(as.formula(rf.formula), data = train.data, importance = TRUE, ntree = 500)
rf
#Predicting test data and calculate sum of square error
test.pred <- predict(rf, test.data)
test.pred
sum((test.pred - log(test.data$SalePrice)) ^ 2)
#--------------------------------------------------------------------------------------------------------------------------------
#Fitting with xgboosting
library(xgboost)
train.label = log(train.data$SalePrice)
test.label = log(test.data$SalePrice)

##gbt <- xgboost(data = train.data[, -dim(train.data)[2]],
##               label = train.label, max_depth = 8,
##               nrounds = 20, objective = 'reg:linear',
##               verbose = 1)

feature.matrix <- model.matrix(~., data = train.data[, - dim(train.data)[2]])
dim(feature.matrix)

set.seed(100)
gbt <- xgboost(data = feature.matrix,
               label = train.label, max_depth = 8,
               nrounds = 20, objective = 'reg:linear',
               verbose = 1)
               
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = gbt)
dim(importance)
sum(importance$Gain)

par <- list(max_depth = 8,
            objecive = 'reg:linear',
            verbose = 1)
gbt.cv <- xgb.cv(params = par,
                 data = feature.matrix,
                 label = train.label,
                 nfold = 5,
                 nrounds = 100)
