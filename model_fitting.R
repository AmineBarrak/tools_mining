#~ install.packages("cvTools")
#~ install.packages("ROSE")
library(cvTools)
library(ROSE)

# Set random seed
# PLEASE SET A SEED NUMBER HERE
set.seed(0)

# Initialise variables
model = 'randomForest'
doVIF = 'YES'
tree_number = 100
target= 'pics'

# Changed types' entropy
df <- as.data.frame(read.csv(file='/home-students/ambard/old/script_r/all_dataset_fix_metrics'), header=TRUE)


df$pics = as.factor(df$pics)


xcol <-c('is_fix','msg_size','author_experience','nbr_insert_line_commit','nbr_deleted_line_commit','nbr_churn_file_commit','MaxNesting','Cyclomatic','CountStmtDecl','CountLineBlank_Php','RatioCommentToCode')
# ,'week_day','month_day','month'
formula <- as.formula(paste('pics ~ ', paste(xcol, collapse= '+')))

#	VIF analysis
if(doVIF == 'YES') {
  library(car)
  formula <- as.formula(paste('pics ~ ', paste(xcol, collapse= '+')))
  fit <- glm(formula, data = df, family = binomial())
  print(vif(fit))
}

# Separate data into k folds
k <- 10
folds <- cvFolds(nrow(df), K = k, type = 'random')

# Initialise result values
acc.vec = pre.vec = rec.vec = fm.vec = npre.vec = nrec.vec = nfm.vec <- c()

#	Iteratively run validation
for(i in 1:k) {
  print(sprintf('Validation no. %d', i))
  #	Split training and testing data
  trainIndex <- folds$subsets[folds$which != i]	# Extract training index
  testIndex <- folds$subsets[folds$which == i]	# Extract testing index			
  trainset <- df[trainIndex, ] 		# Set the training set
  testset <- df[testIndex, ] 			# Set the validation set


  if(model == 'C50') {
    library(C50)
    #i need help here
    #fit <- C5.0(formula, data = train.balanced, rules = TRUE)
    fit = C5.0(formula, data=trainset, rules=T)
    testset[, 'predict'] <- predict(fit,newdata = testset)
    #print(C5imp(fit))
  } else if(model == 'randomForest') {
    library(randomForest)		
    fit <- randomForest(formula, data = trainset, ntree = tree_number, mtry = 5, importance = TRUE)
    testset[, 'predict'] <- predict(fit,newdata = testset)
    #varImpPlot(fit, cex = 1, main = '')
  } else if(model == 'bayes') {
    #i need help here
    library(e1071)
    fit <- naiveBayes(formula, data=trainset)
    
    testset[, 'predict'] <- as.vector(predict(fit, newdata = testset))
  } else if(model == 'glm') {
    fit <- glm(formula, data = trainset, family = 'binomial')
    testset[, 'predict'] <- as.vector(ifelse(predict(fit, type = "response",newdata = testset) > 0.5 , "1", "0"))
  }
  t <- table(observed = testset[, target], predicted = testset[, 'predict'])
  
  actualYES <- testset[testset[target] == '1', ]
  actualNO <- testset[testset[target] == '0', ]
  if(model == 'glm'){
    threshold = 0.5
    tp <- nrow(actualYES[actualYES[,'predict'] > threshold,])
    tn <- nrow(actualNO[actualNO[, 'predict'] <= threshold,])
    fp <- nrow(actualNO[actualNO[, 'predict'] > threshold,])
    fn <-  nrow(actualYES[actualYES[,'predict'] <= threshold,])
  } else {
    tp <- nrow(actualYES[actualYES[,'predict'] == '1',])
    tn <- nrow(actualNO[actualNO[, 'predict'] == '0',])
    fp <- nrow(actualNO[actualNO[, 'predict'] == '1',])
    fn <- nrow(actualYES[actualYES[,'predict'] == '0',])
  }
  
  pre <- tp/(tp+fp)
  rec <- tp/(tp+fn)
  npre <- tn/(tn+fn)
  nrec <- tn/(tn+fp)
  acc.vec <- c(acc.vec, (tn+tp)/(tn+fp+fn+tp))
  pre.vec <- c(pre.vec, pre)
  rec.vec <- c(rec.vec, rec)
  fm.vec <- c(fm.vec, 2*pre*rec/(pre+rec))
  npre.vec <- c(npre.vec, npre)
  nrec.vec <- c(nrec.vec, nrec)
  nfm.vec <- c(nfm.vec, 2*npre*nrec/(npre+nrec))

}

print(sprintf('accuracy: %.1f%%', median(acc.vec) * 100))
print(sprintf('pre: %.1f%%', median(pre.vec) * 100))
print(sprintf('rec: %.1f%%', median(rec.vec) * 100))
print(sprintf('f-measure: %.1f%%', median(fm.vec) * 100))

# output false positives and false negatives in the prediction
write.table(actualYES[actualYES[,'predict'] == '0',], '/home-students/ambard/old/script_r/false/false_positive.csv', row.names = FALSE, col.names=TRUE, sep = ',')
write.table(actualNO[actualNO[,'predict'] == '1',], '/home-students/ambard/old/script_r/false/false_negative.csv', row.names = FALSE, col.names=TRUE, sep = ',')

