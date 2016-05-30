#install.packages("jsonlite")
#install.packages("randomForest")
#install.packages("caret")
#install.packages("e1071")
# install.packages("openNLP") ## Installs the required natural language processing (NLP) package
# #install.packages("openNLPmodels.en") ## Installs the model files for the English language
# install.packages("quanteda")
# library(openNLP) ## Loads the package for use in the task
# #library(openNLPmodels.en) ## 
#install.packages("gbm")

library(quanteda)
library(jsonlite)
library(randomForest)
library(caret)
library(e1071)
library(gbm)
library(stringr)
library(e1071)
library(kernlab)
library(earth)
library(nnet)
library(caret)
library(neuralnet)
setwd("/Users/avneet/Desktop/CS_246-Web Info Mgmt/YELP-PROJECT")

#### FUNCTIONS ###########

calc_score <- function(test.solutions, P1, P2, P3) {
  sum = 0
  for (i in 1:nrow(test.solutions)) {
    actual_score <- (test.solutions$V1[i] + (2*test.solutions$V2[i]) + (5 * test.solutions$V3[i])) + 1
    predicted_score <- (round(P1[i]) + (2*round(P2[i])) + (5 * round(P3[i]))) + 1
    #predicted_score <- ((P1[i]) + (2*P2[i]) + (5 * P3[i])) + 1
    sum = sum + (log(actual_score) - log(predicted_score))^2
  }
  final_score <- sum/nrow(test.solutions)
  #print(final_score)
  s<- sqrt(final_score)
  s
}


### GET DATA AND PREPROCESS ###################

# Fetch Reviews
reviews <- stream_in(file("yelp_boston_academic_dataset/yelp_academic_dataset_review.json"))

# Fetch Business 
business <- stream_in(file("yelp_boston_academic_dataset/yelp_academic_dataset_business.json"))
business <- business[-c(2,3,4,6,9,10,11,13,14,15)]


#Remove Review_id and Type
reviews <- reviews[-c(3,7)]

# Fetch Restaurant_id to Yelp_id mapping
mapping <- read.csv("restaurant_ids_to_yelp_ids.csv", header = TRUE, blank.lines.skip = TRUE)

# Fetch violations
violations <- read.csv("AllViolations.csv", header = TRUE, blank.lines.skip = TRUE,colClasses=c("NULL",NA,NA,NA,NA,NA))
colnames(violations)[1] <- "Date"
colnames(violations)[3:5] <- c("V1","V2","V3")






#Map yelp_id_0
index <- as.character(mapping$yelp_id_0)
names(index) <- as.character(mapping$restaurant_id)

# Convert Date from Factor into Date in Violations and Reviews

violations$Date <- as.Date(violations$Date, format = "%Y-%m-%d")
reviews$date <- as.Date(reviews$date, format = "%Y-%m-%d")


# Iterate through training set to find all reviews for that restaurant before Inspection date.
avg_star_column <- list()
total_reviews <- list()
review_length <- list()
business_ids <- list()
business_rating <- list()
review_chars <- list()
#num_of_categories <- list()
#upper_chars <- list()


for (i in 1:nrow(violations)) {
  print(i)
  #Fetch the appropriate yelp id's based on the restaurant id's
  business_id <- index[violations$restaurant_id[i]][[1]]
  business_ids[i] <- business_id
  review_subset <- reviews[reviews$date < violations$Date[i] & reviews$business_id == business_id,]
  total_reviews[i] <- nrow(review_subset)
  business_rating[i] <- business[business$business_id == business_id,"stars"]
  #num_of_categories[i] <- length(unlist(business[business$business_id == business_id,"categories"]))
  
  if(nrow(review_subset) > 0) {
    avg_star_column[i] <- mean(review_subset$stars)
    l <- sapply(review_subset$text,function(x) nsentence(x), simplify=F)
    review_length[i] <- Reduce('+',l)
    
    m <- sapply(review_subset$text, function(x) nchar(x), simplify = F)
    review_chars[i] <- Reduce('+', m)
    
    #p <- sapply(regmatches(review_subset$text, gregexpr("[A-Z]", review_subset$text, perl=TRUE)), length)
    #upper_chars[i] <- Reduce('+', p)
  } else {
    avg_star_column[i] <- 2
    review_length[i] <- 0
    review_chars[i] <- 0
    #upper_chars[i] <- 0
    #Can experiment with above imputed value
  }  
}



cuisines <- read.csv("cuisine_one_hot_encoding.csv")

## PLAIN VIOLATIONS ####
violations["business_id"] <- unlist(business_ids)
violations["avg1"] <- unlist(avg_star_column)
violations["total_reviews"] <- unlist(total_reviews)
violations["review_length"] <- unlist(review_length)
violations["rating"] <- unlist(business_rating)
violations["diff"] <- abs(violations$rating - violations$avg1)
violations["review_chars"] <- unlist(review_chars)
#violations["num_of_categories"] <- unlist(num_of_categories)
#violations["upper_chars"] <- unlist(upper_chars)

## MERGED VIOLATIONS ####
violations <- merge(violations,cuisines)

#Trying Box Cox transformations##
#A<- BoxCoxTrans(violations$avg1)
#No other variable required trannformation####

# Split violations into training and test data.

set.seed(104572015)
n = nrow(violations)

training.sample.size = 0.75 * n  ###### Use 75% of the data for the training set

training.row.ids = sample( (1:n), training.sample.size )

my.training.set <- violations[  training.row.ids, ]
training.set.size <- nrow(my.training.set)
my.test.set     <- violations[ -training.row.ids, ]   # set complement of training.set.ids
test.set.size <- nrow(my.test.set)




train_predictors <- my.training.set[-c(1:6)]
ctrl <- trainControl(method = "cv", number = 10)
test_predictors <- my.test.set[-c(1:6)]
mset1 <- my.training.set[-c(1,2,3,5,6)]
mset2 <- my.training.set[-c(1,2,3,4,6)]
mset3 <- my.training.set[-c(1,2,3,4,5)]
tset1 <- my.test.set[-c(1,2,3,5,6)]
tset2 <- my.test.set[-c(1,2,3,4,6)]
tset3 <- my.test.set[-c(1,2,3,4,5)]

### MODELS ######
#1.LINEAR REGRESSION WITH 10 FOLD CV :: 1.212933

fit1 <- train(V1 ~ ., data = mset1,
              method = "lm",
              trControl = trainControl(method = "cv"))
P1 <-  predict(fit1, tset1)

fit2 <- train(V2 ~ ., data = mset2,
              method = "lm",
              trControl = trainControl(method = "cv"))
P2 <-  predict(fit2, tset2)

fit3 <- train(V3 ~ ., data = mset3,
              method = "lm",
              trControl = trainControl(method = "cv"))
P3 <-  predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

#2.RANDOM FOREST  WITH 2 FOLD CV :: 1.119508991

fit1 <- train(V1 ~ ., data = mset1,
              method = "rf",
              #ntree = 1500,
              trControl = trainControl(method = "cv",number = 2))
P1 <-  predict(fit1, tset1)

fit2 <- train(V2 ~ ., data = mset2,
              method = "rf",
              #ntree = 1500,
              trControl = trainControl(method = "cv",number = 2))
P2 <-  predict(fit2, tset2)

fit3 <- train(V3 ~ ., data = mset3,
              method = "rf",
              #ntree = 1500,
              trControl = trainControl(method = "cv",number = 2))
P3 <-  predict(fit3, tset3)
ans<-calc_score(my.test.set, P1,P2,P3)

##3.NEURAL NETWORK INCONCLUSIVE

fit1 <- train(V1 ~ ., data = mset1,
              method = "neuralnet",
              trControl = trainControl(method = "cv",number=2))
P1 <-  predict(fit1, tset1)

fit2 <- train(V2 ~ ., data = mset2,
              method = "neuralnet",
              trControl = trainControl(method = "cv",number=2))
P2 <-  predict(fit2, tset2)

fit3 <- train(V3 ~ ., data = mset3,
              method = "neuralnet",
              trControl = trainControl(method = "cv",number=2))
P3 <-  predict(fit3, tset3)

## 4. KNN MODEL ####
fit1 <- train(V1 ~.,
              data = mset1,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:10),
              trControl = trainControl(method = "cv", number = 2))
P1 <- predict(fit1, tset1)

fit2 <- train(V2 ~.,
              data = mset2,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:10),
              trControl = trainControl(method = "cv", number = 2))
P2 <- predict(fit2, tset2)

fit3 <- train(V3 ~.,
              data = mset3,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:10),
              trControl = trainControl(method = "cv", number = 2))
P3 <- predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)
















#Linear Regression model: ### 1.213315 ###
# fit1 <- train(x = train_predictors, y = my.training.set$V1, method = "lm"
#               #, trControl = ctrl
#               )
fit1 <- lm(V1 ~ ., data = mset1)
P1 <-  predict(fit1, tset1)
# # 
# fit2 <- train(x = train_predictors, y = my.training.set$V2, method = "lm"
#               #, trControl = ctrl
#               )
fit2 <- lm(V2 ~ ., data = mset2)
P2 <-  predict(fit2, tset2)
# # 
# fit3 <- train(x = train_predictors, y = my.training.set$V3,method = "lm"
#               #, trControl = ctrl
#               )
fit3 <- lm(V3 ~ ., data = mset3)
P3 <-  predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

#Random Forest model ## 1.117 ###/1.108//# 1.034375/1.119 with all/1.117838 with originals
fit1 <- randomForest(V1 ~ ., mset1 , ntree=1000)
P1 <-  predict(fit1, tset1)
# # 
fit2 <- randomForest(V2 ~ ., mset2 , ntree=1000)
P2 <-  predict(fit2, tset2)
# # 
fit3 <- randomForest(V3 ~ . , mset3 , ntree=1000)
P3 <-  predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

#Neural Network Model ### 

fit1 <- nnet(V1 ~ ., mset1, size = 5
             #, decay = 0.01, linout = TRUE, trace = FALSE
)
P1 <- predict(fit1, tset1)

fit2 <- nnet(V2 ~ ., mset2, size = 5
             #, size = 5, decay = 0.01, linout = TRUE, trace = FALSE
)
P2 <- predict(fit2, tset2)

fit3 <- nnet(V3 ~ ., mset3 , size =5
             #, size = 5, decay = 0.01, linout = TRUE, trace = FALSE
)
P3 <- predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

#MARS MODEL ## 
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
fit1 <- train(V1 ~ .,
              data = mset1,
              #x = train_predictors, y = train_solu$solubility, 
              method = "earth"
              ##preProc = c("scale"),
              #tuneGrid = marsGrid
              #trControl = trainControl(method= "repeatedcv",repeats = 3)
)
P1 <- predict(fit1, tset1)

fit2 <- train(V2 ~ .,
              data = mset2,
              #x = train_predictors, y = train_solu$solubility, 
              method = "earth"
              #,#preProc = c("scale"),
              #tuneGrid = marsGrid
              #trControl = trainControl(method= "repeatedcv",repeats = 3)
)
P2 <- predict(fit2, tset2)

fit3 <- train(V3 ~ .,
              data = mset3,
              #x = train_predictors, y = train_solu$solubility, 
              method = "earth"
              #,#preProc = c( "scale"),
              #tuneGrid = marsGrid
              #trControl = trainControl(method= "repeatedcv",repeats = 3)
)
P3 <- predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

## KNN Model #####   
fit1 <- train(V1 ~.,
              data = mset1,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:30),
              trControl = trainControl(method = "cv"))
P1 <- predict(fit1, tset1)

fit2 <- train(V2 ~.,
              data = mset2,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:30),
              trControl = trainControl(method = "cv"))
P2 <- predict(fit2, tset2)

fit3 <- train(V3 ~.,
              data = mset3,
              method = "knn",
              #preProc = c("center", "scale"),
              tuneGrid = data.frame(.k = 1:30),
              trControl = trainControl(method = "cv"))
P3 <- predict(fit3, tset3)

ans<-calc_score(my.test.set, P1,P2,P3)

### gbm model ###
mset <- my.training.set[-c(1,2,3,5,6)]
#fit1 <- lm(V1 ~ review_length , my.training.set)
fit1 <- gbm(V1 ~ ., data = mset , n.trees=100,distribution = "gaussian")
tset <- my.test.set[-c(1,2,3,5,6)]
P1 <-  predict(fit1, tset,n.trees=2000)
# # 
mset <- my.training.set[-c(1,2,3,4,6)]
#fit2 <- lm(V2 ~ review_length , my.training.set)
fit2 <- gbm(V2 ~ ., data = mset , n.trees=100, distribution = "gaussian")
tset <- my.test.set[-c(1,2,3,4,6)]
P2 <-  predict(fit2, tset, n.trees = 2000)
# # 
mset <- my.training.set[-c(1,2,3,4,5)]
#fit3 <- lm(V3 ~ review_length, my.training.set)
fit3 <- gbm(V3 ~ . , data = mset , n.trees=100, distribution = "gaussian")
tset <- my.test.set[-c(1,2,3,4,5)]
P3 <-  predict(fit3, tset, n.trees = 2000)

ans<-calc_score(my.test.set, P1,P2,P3) 

# 1.2 using  cuisines with RF tree 500
# 1.171 using total_reviews + cuisines with RF tree 500
# 1.124849 using total_reviews + cuisines+avg rating with RF tree 700 *****
# 1.111374 using total_reviews + cuisines+avg rating + length of review with RF tree 1000 *****
# 1.110001 using total_reviews + cuisines+avg rating + length of review + business rating
# 1.094424 using total_reviews + cuisines+avg rating + length of review + business rating +(avg-business rating) with 1000 trees***




# MODEL FITTING AND PREDICTION

#Model 1 - with avg stars . RMSE = 1.091608 using lm and 1.109473 using Random Forest(with imputing 2)

#fit1 <- lm(V1 ~ avg1 , my.training.set)
# fit1 <- randomForest(V1 ~ rating , my.training.set , ntree=50)
# P1 <-  predict(fit1, my.test.set)
# 
# #fit2 <- lm(V2 ~ avg1 , my.training.set)
# fit2 <- randomForest(V2 ~ rating , my.training.set , ntree=50)
# P2 <-  predict(fit2, my.test.set)
# 
# #fit3 <- lm(V3 ~ avg1 , my.training.set)
# fit3 <- randomForest(V3 ~ rating , my.training.set , ntree=50)
# P3 <-  predict(fit3, my.test.set)

#Model 2 - with review count . RMSE = 1.091225 using lm and 1.106112 using RandomForest.On whole set, 1.23

# fit1 <- lm(V1 ~ review_chars , my.training.set)
# #fit1 <- randomForest(V1 ~ total_reviews , my.training.set , ntree=50)
# P1 <-  predict(fit1, my.test.set)
# 
# fit2 <- lm(V2 ~ review_chars , my.training.set)
# #fit2 <- randomForest(V2 ~ total_reviews , my.training.set , ntree=50)
# P2 <-  predict(fit2, my.test.set)
# 
# fit3 <- lm(V3 ~ review_chars , my.training.set)
# #fit3 <- randomForest(V3 ~ total_reviews , my.training.set , ntree=50)
# P3 <-  predict(fit3, my.test.set)


#Model 3 - with review_length count . RMSE = 1.119809 using lm and 1.123109 by Random Forest.

# fit1 <- lm(V1 ~ log10(review_length+1) , my.training.set)
# # fit1 <- randomForest(V1 ~ review_length , my.training.set , ntree=50)
# P1 <-  predict(fit1, my.test.set)
# # # 
# fit2 <- lm(V2 ~ log10(review_length+1) , my.training.set)
# # fit2 <- randomForest(V2 ~ review_length , my.training.set , ntree=50)
# P2 <-  predict(fit2, my.test.set)
# # # 
# fit3 <- lm(V3 ~ log10(review_length+1) , my.training.set)
# # fit3 <- randomForest(V3 ~ review_length , my.training.set , ntree=50)
# P3 <-  predict(fit3, my.test.set)
# # # 

#Model 4 - with starts + review_count . RMSE =  1.11981

# fit1 <- lm(V1 ~ avg1 + total_reviews , my.training.set)
# #fit1 <- randomForest(V1 ~ avg1 + total_reviews , my.training.set , ntree=50)
# P1 <-  predict(fit1, my.test.set)
# # 
# fit2 <- lm(V2 ~ avg1 + total_reviews , my.training.set)
# #fit2 <- randomForest(V2 ~ avg1 + total_reviews , my.training.set , ntree=50)
# P2 <-  predict(fit2, my.test.set)
# # 
# fit3 <- lm(V3 ~ avg1 + total_reviews , my.training.set)
# #fit3 <- randomForest(V3 ~ avg1 + total_reviews , my.training.set , ntree=50)
# P3 <-  predict(fit3, my.test.set)
# #

#Model 5 - Combination.RMSE = 

#fit1 <- lm(V1 ~ review_length , my.training.set)
# fit1 <- randomForest(V1 ~ total_reviews + review_length + avg1 , my.training.set , ntree=500)
# P1 <-  predict(fit1, my.test.set)
# # # 
# #fit2 <- lm(V2 ~ review_length , my.training.set)
# fit2 <- randomForest(V2 ~ total_reviews + review_length + avg1 , my.training.set , ntree=500)
# P2 <-  predict(fit2, my.test.set)
# # # 
# #fit3 <- lm(V3 ~ review_length, my.training.set)
# fit3 <- randomForest(V3 ~ total_reviews + review_length + avg1 , my.training.set , ntree=500)
# P3 <-  predict(fit3, my.test.set)


# 1.231937 using review_length and lm and 1.23733 using RF.
# 1.223823 using total_reviews
# 1.198903 using total_reviews + reviews_length with RF tree 500
# 1.166651 using total_reviews + reviews_length +avg1 with RF tree 700


## Code to create cusine file #####
# r <- reviews[-c(1,3,7)]
# r["funny_votes"] <- reviews$votes$funny
# r["useful_votes"] <- reviews$votes$useful
# r["cool_votes"] <- reviews$votes$cool
# r$text <- trimws(r$text)
# r$text <- gsub("[\r\n]", "", r$text)
# write.table(r, "reviews.txt",sep="|",col.names = TRUE)
ans