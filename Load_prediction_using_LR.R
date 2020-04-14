# Loan prediction Model

# Importing the dataset
trainData = read.csv('train_data.csv')
testData = read.csv('test_data.csv')
submissionData = read.csv('sample_submission.csv')
trainData = trainData[, -1]
testData = testData[, -1]


#Encoding the categorical data of the training set into factors
trainData$Gender = factor(trainData$Gender,
                          levels = c('Male', 'Female'),
                          labels = c(1, 2))

trainData$Married = factor(trainData$Married,
                           levels = c('Yes', 'No'),
                           labels = c(1, 0))

trainData$Education = factor(trainData$Education,
                             levels = c('Graduate', 'Not Graduate'),
                             labels = c(1, 0))

trainData$Self_Employed = factor(trainData$Self_Employed,
                                 levels = c('Yes', 'No'),
                                 labels = c(1, 0))

trainData$Property_Area = factor(trainData$Property_Area,
                                 levels = c('Urban', 'Semiurban', 'Rural'),
                                 labels = c(1, 2, 3))

trainData$Loan_Status = factor(trainData$Loan_Status,
                               levels = c('Y', 'N'),
                               labels = c(1, 0))



#Encoding the categorical data of the test set into factors
testData$Gender = factor(testData$Gender,
                         levels = c('Male', 'Female'),
                         labels = c(1, 2))

testData$Married = factor(testData$Married,
                          levels = c('Yes', 'No'),
                          labels = c(1, 0))

testData$Education = factor(testData$Education,
                            levels = c('Graduate', 'Not Graduate'),
                            labels = c(1, 0))

testData$Self_Employed = factor(testData$Self_Employed,
                                levels = c('Yes', 'No'),
                                labels = c(1, 0))

testData$Property_Area = factor(testData$Property_Area,
                                levels = c('Urban', 'Semiurban', 'Rural'),
                                labels = c(1, 2, 3))


# Handling the Categorical Missing Data of the training set

Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}

i1 = c(1, 2, 5, 10)

trainData[i1] <- lapply(trainData[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))


#Handling the Numrical Missing data of the training set
trainData$LoanAmount = ifelse(is.na(trainData$LoanAmount),
                              ave(trainData$LoanAmount, FUN = function(x) mean(x, na.rm = TRUE)),
                              trainData$LoanAmount)

trainData$Loan_Amount_Term = ifelse(is.na(trainData$Loan_Amount_Term),
                                    ave(trainData$Loan_Amount_Term, FUN = function(x) mean(x, na.rm = TRUE)),
                                    trainData$Loan_Amount_Term)


# Handling the Categorical Missing Data of the test set


testData[i1] <- lapply(testData[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))


#Handling the Numrical Missing data of the test set
testData$LoanAmount = ifelse(is.na(testData$LoanAmount),
                             ave(testData$LoanAmount, FUN = function(x) mean(x, na.rm = TRUE)),
                             testData$LoanAmount)

testData$Loan_Amount_Term = ifelse(is.na(testData$Loan_Amount_Term),
                                   ave(testData$Loan_Amount_Term, FUN = function(x) mean(x, na.rm = TRUE)),
                                   testData$Loan_Amount_Term)


# Feature Scaling
trainData[, 6:9] = scale(trainData[, 6:9])
testData[, 6:9] = scale(testData[, 6:9])


#Fitting classifier to the Training set
classifier = glm(formula = Loan_Status ~ .,
                 family = binomial,
                 data = trainData)

# Applying k-fold cross validation to our model to check its accuracy
library(caret)
folds = createFolds(trainData$Loan_Status, k=10)
cv = lapply(folds, function(x) {
  training_fold = trainData[-x, ]
  test_fold = trainData[x, ]
  
  #Fitting classifier to the Training set
  classifier = glm(formula = Loan_Status ~ .,
                   family = binomial,
                   data = training_fold)
  
  prob_pred = predict(classifier, type = 'response', newdata = test_fold[-12])
  y_pred = ifelse(prob_pred > 0.5, 1, 0)
  
  
  cm = table(test_fold[, 12], y_pred)
  
  accuracy = (cm[1,1] + cm[2,2])/(cm[1,1] + cm[2,2] + cm[2,1] + cm[1,2])
  
  return(accuracy)
  
})

#Checking overall accuracy 
ave_accuracy = mean(as.numeric(cv))




# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = testData)
y_pred = ifelse(prob_pred > 0.5, 0, 1)



