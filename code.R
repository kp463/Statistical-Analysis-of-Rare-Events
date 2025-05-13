rm(list=ls())
set.seed(2207)

# loading libraries----------------------------
library(ggplot2)
library(logistf)
library(brglm2)
library(glmnet)
library(patchwork)
library(dplyr)
library(zoo)
library(caret)
library(pROC)
library(tree)
library(randomForest)
library(rpart)
library(ipred)

# loading dataset------------------------------
setwd("D:/ISI/Sem-4/Project")
data=read.csv(file="data2.csv")
data
nrow(data)

# Data Cleaning--------------------------------
summary(data[,1])
table(data[,1])

summary(data[,4])
table(data[,4])

data=data[-which(data$person_age>100|data$person_emp_length>100),]
nrow(data)

# EDA------------------------------------------

#Graphical approach------------

## age----------
any(is.na(data[,1]))  # no missing value
table(data[,1])

p1=ggplot(data.frame(x = data[,1]), aes(x)) +
  geom_histogram(aes(y = ..density..),binwidth = 0.5, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "Age",
       y = "Density") 

Group=as.factor(data[,9])
q1=ggplot(data, aes(x = Group, y = data[,1], group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "age")


## person income------------
any(is.na(data[,2]))  # no missing value

table(data[,2])

p2=ggplot(data.frame(x = data[,2])/1000, aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "Income ('000)",
       y = "Density") 

q2=ggplot(data, aes(x = Group, y = data[,2]/1000, group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "income('000)")+
  ylim(c(0,200))

## ownership------------------
any(is.na(data[,3]))  # no missing value

freq_table=table(data[,3])
freq_table
freq_df <- data.frame(categories = names(freq_table), frequency = as.numeric(freq_table))

p3=ggplot(freq_df, aes(x = categories, y = frequency/sum(frequency))) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(x = "personal home ownership",
       y = "proportion") 

q3=ggplot(data, aes(x = data[,3], fill = as.factor(data[,9]))) +
  geom_bar(position = "dodge", color = "black") +
  labs(x = "personal home ownership",
       y = "Count",fill="loan status")

## employment length--------------------
any(is.na(data[,4]))  # missing value
m1=which(is.na(data[,4]))
length(which(is.na(data[,4])))

table(data[,4])

# Impute missing values with mode by group----
d <- data.frame(group = data[,1],
                value = data[,4])
imputed_data <- d %>%
  group_by(group) %>%
  mutate(value = na.aggregate(value, FUN = function(x) {
    m <- table(x)
    as.numeric(names(m)[which.max(m)])
  }))
data[,4]=as.vector(imputed_data[,2])
any(is.na(data[,4]))

p4=ggplot(data.frame(x = data[,4]), aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "emp length",
       y = "Density")

q4=ggplot(data, aes(x = Group, y = data[,4], group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "emp length")+
  ylim(c(0,20))

## lone intent--------------------
any(is.na(data[,5]))  # no missing value
freq_table=table(data[,5])
freq_table
freq_df <- data.frame(categories = names(freq_table), frequency = as.numeric(freq_table))

p5=ggplot(freq_df, aes(x = categories, y = frequency/sum(frequency))) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(x = "lone intent",
       y = "proportion") 

q5=ggplot(data, aes(x = data[,5], fill = as.factor(data[,9]))) +
  geom_bar(position = "dodge", color = "black") +
  labs(x = "lone intent",
       y = "Count",
       fill="loan status") 

## loan grade-----------------------
any(is.na(data[,6]))  # no missing value
freq_table=table(data[,6])
freq_table
freq_df <- data.frame(categories = names(freq_table), frequency = as.numeric(freq_table))

p6=ggplot(freq_df, aes(x = categories, y = frequency/sum(frequency))) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(x = "lone grade",
       y = "proportion") 

q6=ggplot(data, aes(x = data[,6], fill = as.factor(data[,9]))) +
  geom_bar(position = "dodge", color = "black") +
  labs(x = "lone grade",
       y = "Count",
       fill="loan status") 

## loan amount---------------------
any(is.na(data[,7]))  # no missing value
table(data[,7])

p7=ggplot(data.frame(x = data[,7]/1000), aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "loan amount('000)",
       y = "Density")

q7=ggplot(data, aes(x = Group, y = data[,7]/1000, group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "loan amount('000)")+
  ylim(c(0,25))

## interest rate---------------------
any(is.na(data[,8]))  # missing value
m1=which(is.na(data[,8]))
length(m1)

summary(data[,8])
table(data[,8])

# Impute missing values with mode by group----
d <- data.frame(group = data[,7],
                value = data[,8])
compute_mode <- function(x) {
  tab <- table(x)
  ifelse(length(tab) == 0, 11.01, as.numeric(names(tab)[which.max(tab)]))
}

# Impute missing values with mode by group
imputed_data <- d %>%
  group_by(group) %>%
  mutate(value = ifelse(is.na(value), compute_mode(value), value))

data[,8]=as.vector(imputed_data[,2])
any(is.na(data[,8]))

p8=ggplot(data.frame(x = data[,8]), aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "interest rate",
       y = "Density")

q8=ggplot(data, aes(x = Group, y = data[,8], group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "interest rate")

## loan percent income------------
any(is.na(data[,10]))  # no missing value

p9=ggplot(data.frame(x = data[,10]), aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(x = "loan percent income",
       y = "Density")

q9=ggplot(data, aes(x = Group, y = data[,10], group = Group)) +
  geom_boxplot(fill="red") +
  labs(x = "loan status", y = "lone percent income")

## Historical Default-----------------------
any(is.na(data[,11]))  # no missing value
freq_table=table(data[,11])
freq_table
freq_df <- data.frame(categories = names(freq_table), frequency = as.numeric(freq_table))

p10=ggplot(freq_df, aes(x = categories, y = frequency/sum(frequency))) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(x = "default",
       y = "proportion") 

q10=ggplot(data, aes(x = data[,11], fill = as.factor(data[,9]))) +
  geom_bar(position = "dodge", color = "black") +
  labs(x = "Default",
       y = "Count",
       fill="loan status") 

## length-----------------------------
any(is.na(data[,12]))  # no missing value
freq_table=table(data[,12])
freq_table

ggplot(data.frame(x = data[,12]), aes(x)) +
  geom_histogram(aes(y = ..density..),bins=30, fill = "skyblue", 
                 color = "black", alpha = 0.7) +
  labs(title = "Distribution of Length ",
       x = "length",
       y = "Density")

ggplot(data, aes(x = Group, y = data[,12], group = Group)) +
  geom_boxplot(fill="red") +
  labs(title = "Groupwise Boxplot", x = "loan status", y = "length")

p1+p2+p4+p7+p8+p9
p3+p5+p6+p10
q1+q2+q4+q7+q8+q9
q3+q5+q6+q10

# Independence test-----------------------------

## Ctaegorical variables---------

tab1=table(data[,3],data[,9])
tab1
chisq.test(tab1)  # home ownership is significant

tab2=table(data[,5],data[,9])
tab2
chisq.test(tab2)  # lone intent is significant

tab3=table(data[,6],data[,9])
tab3
chisq.test(tab3)  # lone grade is significant

tab4=table(data[,11],data[,9])
tab4
chisq.test(tab4)  # default is significant


# Setting the dataset------------------------------------------------

data=data[,-c(10,12)]  # Removing the last variable
ncol(data)

data[,6]
data[,6]=ifelse(data[,6]=="A"|data[,6]=="B"|data[,6]=="C","Good","Bad")

attach(data)
library(fastDummies)
data=dummy_cols(data,select_columns = c("person_home_ownership","loan_intent",
                                        "loan_grade","cb_person_default_on_file"),
                remove_first_dummy = T,
                remove_selected_columns = T)
View(data)
ncol(data)


# Breaking the dataset in train and test set-------------------------

train_size <- round(0.8 * nrow(data))
train_indices = createDataPartition(data$loan_status,p=0.8)$Resample1

# Create training and testing sets
train_set = data[train_indices, ]
test_set = data[-train_indices, ]
nrow(train_set)
nrow(test_set)

table(train_set$loan_status)
actual=test_set[,6]
test_set=test_set[,-6]

cutoff=seq(0.001,0.999,0.001)
length(cutoff)

# Logistic Regression Model-------------------------------------------
model1 <- glm(loan_status~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
table(actual,predicted)

# Firth Logistic Model-----------------------
model2 <- logistf(loan_status~.,data = train_set)
summary(model2)

prob2=predict(model2,newdata = test_set,type = "response")
length(prob2)

mis_error2=array(0)
for(i in 1:999)
{
  predicted=ifelse(prob2>cutoff[i],1,0)
  mis_error2[i]=length(actual[actual!=predicted])/length(actual)
}

error2=data.frame(cutoff,mis_error2)
ggplot(error2, aes(x = cutoff, y = mis_error2)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs",
       subtitle = "Firth Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")

sen2=array(0)
prec2=array(0)
for(i in 1:995)
{
  predicted=ifelse(prob2>cutoff[i],1,0)
  s=table(predicted,actual)
  sen2[i]=s[2,2]/(s[1,2]+s[2,2])
  prec2[i]=s[2,2]/(s[2,1]+s[2,2])
}

sensitivity2=data.frame(cutoff,sen2)
ggplot(sensitivity2, aes(x = cutoff, y = sen2)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Firth Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision2=data.frame(cutoff,prec2)
ggplot(precision2, aes(x = cutoff, y = prec2)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Firth Logistic Regression Model",
       x = "cutoff", y = "Precision")

predicted=ifelse(prob2>0.5,1,0)
mis_error2=length(actual[actual!=predicted])/length(actual)
sen2=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec2=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr2=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc2=roc(actual,prob2)
auc(roc2)

predicted=ifelse(prob2>0.5,1,0)
s=table(predicted,actual)
sen2=s[2,2]/(s[1,2]+s[2,2])
prec2=s[2,2]/(s[2,1]+s[2,2])
table(actual,predicted)
# Firth Logistic Model with Added Covariate Penalty -----------------
model3 <- flac(loan_status~.,data = train_set)
summary(model3)

prob3=predict(model3,newdata = test_set,type = "response")
length(prob3)

mis_error3=array(0)
for(i in 1:999)
{
  predicted=ifelse(prob3>cutoff[i],1,0)
  mis_error3[i]=length(actual[actual!=predicted])/length(actual)
}

error3=data.frame(cutoff,mis_error3)
ggplot(error3, aes(x = cutoff, y = mis_error3)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs",
       subtitle = "Firth Logistic Model with Added Covariate",
       x = "cutoff", y = "Misclassification Error")

sen3=array(0)
prec3=array(0)
for(i in 1:995)
{
  predicted=ifelse(prob3>cutoff[i],1,0)
  s=table(predicted,actual)
  sen3[i]=s[1,2]/(s[1,2]+s[2,2])
  prec3[i]=s[2,2]/(s[2,1]+s[2,2])
}

sensitivity3=data.frame(cutoff,sen3)
ggplot(sensitivity3, aes(x = cutoff, y = sen3)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Firth Logistic Model with Added Covariate",
       x = "cutoff", y = "Sensitivity")

precision3=data.frame(cutoff,prec3)
ggplot(precision3, aes(x = cutoff, y = prec3)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Firth Logistic Regression Model with Added Covariate",
       x = "cutoff", y = "Precision")

predicted=ifelse(prob3>0.5,1,0)
s=table(predicted,actual)
sen3=s[2,2]/(s[1,2]+s[2,2])
prec3=s[2,2]/(s[2,1]+s[2,2])

roc3=roc(actual,prob3)
auc(roc3)
table(actual,predicted)
# Ridge Regression---------------------------------------
model4 = glmnet(x=train_set[,-6], y=train_set[,6], family = "binomial", alpha = 1,
                lambda=0.0003898644)
summary(model4)
coef(model4)
pval(model4)
cv_model <- cv.glmnet(x = as.matrix(train_set[,-6]), y = train_set[,6], 
                      family = "binomial", alpha = 1)
plot(cv_model)
lambda.min = cv_model$lambda.min

prob4=predict(model4,newx = as.matrix(test_set),type = "response")
length(prob4)
mis_error4=array(0)
for(i in 1:995)
{
  predicted=ifelse(prob4>cutoff[i],1,0)
  mis_error4[i]=length(actual[actual!=predicted])/length(actual)
}

error4=data.frame(cutoff,mis_error4)
ggplot(error4, aes(x = cutoff, y = mis_error4)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs",
       subtitle = "Firth Logistic Model with Added Covariate",
       x = "cutoff", y = "Misclassification Error")

sen4=array(0)
prec4=array(0)
for(i in 1:995)
{
  predicted=ifelse(prob4>cutoff[i],1,0)
  s=table(predicted,actual)
  sen4[i]=s[1,2]/(s[1,2]+s[2,2])
  prec4[i]=s[2,2]/(s[2,1]+s[2,2])
}

sensitivity4=data.frame(cutoff,sen4)
ggplot(sensitivity4, aes(x = cutoff, y = sen4)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Ridge Regression",
       x = "cutoff", y = "Sensitivity")

precision4=data.frame(cutoff,prec4)
ggplot(precision4, aes(x = cutoff, y = prec4)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Ridge Regression",
       x = "cutoff", y = "Precision")


# Decision Tree----------------------------------------------
model5=tree(as.factor(loan_status)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
table(actual,predicted)

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])

# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=i) 

predicted=predict(model6,test_set) # Predicting the test set

mis_error6[i]=length(actual[actual!=predicted])/length(actual)
sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=17)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])


# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
  formula = as.factor(loan_status) ~ .,
  data = train_set,
  nbagg = i,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0)
  )

predicted=predict(model7,test_set) # Predicting the test set

mis_error7[i]=length(actual[actual!=predicted])/length(actual)
sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(loan_status) ~ .,
  data = train_set,
  nbagg = 17,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])


df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3

# knn----------------------------------
library(class)
model6 <- knn(train_set[,-6],test_set,train_set$loan_status,k=3)
model6
            
predicted=model6 # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])

# Logistic 2-------------------------------------------------
train_set=train_set[,-c(1,16)]
test_set=test_set[,-c(1,16)]

model1 <- glm(loan_status~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


predicted=ifelse(prob1>0.5,1,0)
s=table(predicted,actual)
sen1=s[2,2]/(s[1,2]+s[2,2])
prec1=s[2,2]/(s[2,1]+s[2,2])

roc1=roc(actual,prob1)
auc(roc1)

# Firth Logistic Model-----------------------
model2 <- logistf(loan_status~.,data = train_set)
summary(model2)

prob2=predict(model2,newdata = test_set,type = "response")
length(prob2)

mis_error2=array(0)
for(i in 1:999)
{
  predicted=ifelse(prob2>cutoff[i],1,0)
  mis_error2[i]=length(actual[actual!=predicted])/length(actual)
}

error2=data.frame(cutoff,mis_error2)
ggplot(error2, aes(x = cutoff, y = mis_error2)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs",
       subtitle = "Firth Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


predicted=ifelse(prob2>0.5,1,0)
s=table(predicted,actual)
sen2=s[2,2]/(s[1,2]+s[2,2])
prec2=s[2,2]/(s[2,1]+s[2,2])

roc2=roc(actual,prob2)
auc(roc2)

# Firth Logistic Model with Added Covariate Penalty -----------------
model3 <- flac(loan_status~.,data = train_set)
summary(model3)

prob3=predict(model3,newdata = test_set,type = "response")
length(prob3)

mis_error3=array(0)
for(i in 1:999)
{
  predicted=ifelse(prob3>cutoff[i],1,0)
  mis_error3[i]=length(actual[actual!=predicted])/length(actual)
}

error3=data.frame(cutoff,mis_error3)
ggplot(error3, aes(x = cutoff, y = mis_error3)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs",
       subtitle = "Firth Logistic Model with Added Covariate",
       x = "cutoff", y = "Misclassification Error")


predicted=ifelse(prob3>0.5,1,0)
s=table(predicted,actual)
sen3=s[2,2]/(s[1,2]+s[2,2])
prec3=s[2,2]/(s[2,1]+s[2,2])


# SMOTE------------------------------------------------------------------
table(data$loan_status)
library(smotefamily)
A=SMOTE(data[,-6],data$loan_status)
B=A$data
head(B)
table(B$class)
View(B)

# Breaking the dataset in train and test set-------------------------
train_size <- round(0.8 * nrow(B))
train_indices = createDataPartition(B$class,p=0.8)$Resample1

# Create training and testing sets--------
train_set = B[train_indices, ]
test_set = B[-train_indices, ]

nrow(train_set)
nrow(test_set)

table(train_set$class)

as.numeric(train_set$class)
actual=as.numeric(test_set$class)
test_set=test_set[,-16]
#knn-----------------------------
library(class)
model6 <- knn(train_set[,-16],test_set,train_set$class,k=5)
model6

predicted=model6 # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])

# Logistic Regression Model-------------------------------------------
model1 <- glm(as.numeric(train_set$class)~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)

table(actual,predicted)

# Firth Logistic Regression Model-------------------------------------------
model2 <- logistf(as.numeric(train_set$class)~.,data = train_set, family = "binomial")
summary(model2)

prob2=predict(model1,newdata = test_set,type = "response")
length(prob2)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)

table(actual,predicted)
# FLAC-------------------------------------------
model3 <- flac(as.numeric(train_set$class)~.,data = train_set, family = "binomial")
summary(model3)

prob3=predict(model1,newdata = test_set,type = "response")
length(prob3)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
table(actual,predicted)

# Decision Tree----------------------------------------------
model5=tree(as.factor(train_set$class)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
predicted

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5
# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=i) 
  
  predicted=predict(model6,test_set) # Predicting the test set
  
  mis_error6[i]=length(actual[actual!=predicted])/length(actual)
  sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=19)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
    formula = as.factor(train_set$class) ~ .,
    data = train_set,
    nbagg = i,   
    coob = FALSE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  predicted=predict(model7,test_set) # Predicting the test set
  
  mis_error7[i]=length(actual[actual!=predicted])/length(actual)
  sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(train_set$class) ~ .,
  data = train_set,
  nbagg = 19,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3

#R3--------------------------------------------------------------
runif(1,5,10)  # event frequency
n=7108-floor((6.96*25467/100))  # no of default obs to remove
n
d=which(data$loan_status==1)  # default index
ind=sample(d,n,replace = FALSE)
data1=data[-ind,]  # R3 data
View(data1)
nrow(data1)
table(data1$loan_status)

# Breaking the dataset in train and test set-------------------------
train_size <- round(0.8 * nrow(data1))
train_indices = createDataPartition(data1$loan_status,p=0.8)$Resample1

# Create training and testing sets
train_set = data1[train_indices, ]
test_set = data1[-train_indices, ]
nrow(train_set)
nrow(test_set)

table(train_set$loan_status)
actual=test_set[,6]
test_set=test_set[,-6]

cutoff=seq(0.001,0.999,0.001)
length(cutoff)

# Logistic Regression Model-------------------------------------------
model1 <- flac(loan_status~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
table(actual,predicted)
1-mis_error1
sen1
prec1
# Decision Tree----------------------------------------------
model5=tree(as.factor(loan_status)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
predicted

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
table(actual,predicted)
1-mis_error5

# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=i) 
  
  predicted=predict(model6,test_set) # Predicting the test set
  
  mis_error6[i]=length(actual[actual!=predicted])/length(actual)
  sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=19)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
    formula = as.factor(loan_status) ~ .,
    data = train_set,
    nbagg = i,   
    coob = FALSE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  predicted=predict(model7,test_set) # Predicting the test set
  
  mis_error7[i]=length(actual[actual!=predicted])/length(actual)
  sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(loan_status) ~ .,
  data = train_set,
  nbagg = 19,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3

# SMOTE------------------------------------------------------------------
table(data1$loan_status)
library(smotefamily)
A=SMOTE(data1[,-6],data1$loan_status)
B=A$data
head(B)
table(B$class)
View(B)

# Breaking the dataset in train and test set-------------------------
train_size <- round(0.8 * nrow(B))
train_indices = createDataPartition(B$class,p=0.8)$Resample1

# Create training and testing sets--------
train_set = B[train_indices, ]
test_set = B[-train_indices, ]

nrow(train_set)
nrow(test_set)

table(train_set$class)

as.numeric(train_set$class)
actual=as.numeric(test_set$class)
test_set=test_set[,-16]

model1 <- flac(as.numeric(train_set$class)~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
1-mis_error1
sen1
prec1
# Decision Tree----------------------------------------------
model5=tree(as.factor(train_set$class)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
predicted

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5
# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=i) 
  
  predicted=predict(model6,test_set) # Predicting the test set
  
  mis_error6[i]=length(actual[actual!=predicted])/length(actual)
  sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=17)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
    formula = as.factor(train_set$class) ~ .,
    data = train_set,
    nbagg = i,   
    coob = FALSE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  predicted=predict(model7,test_set) # Predicting the test set
  
  mis_error7[i]=length(actual[actual!=predicted])/length(actual)
  sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(train_set$class) ~ .,
  data = train_set,
  nbagg = 17,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3

#R2----------------------------------------------------------
runif(1,1,5)  # event frequency
n=7108-floor((3.31*25467/100))  # no of default obs to remove
n
d=which(data$loan_status==1)  # default index
ind=sample(d,n,replace = FALSE)
data1=data[-ind,]  # R3 data
View(data1)
nrow(data1)
table(data1$loan_status)

# Breaking the dataset in train and test set-------------------------
train_size <- round(0.8 * nrow(data1))
train_indices = createDataPartition(data1$loan_status,p=0.8)$Resample1

# Create training and testing sets
train_set = data1[train_indices, ]
test_set = data1[-train_indices, ]
nrow(train_set)
nrow(test_set)

table(train_set$loan_status)
actual=test_set[,6]
test_set=test_set[,-6]

cutoff=seq(0.001,0.999,0.001)
length(cutoff)

# Logistic Regression Model-------------------------------------------
model1 <- flac(loan_status~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
1-mis_error1
# Decision Tree----------------------------------------------
model5=tree(as.factor(loan_status)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
predicted

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
table(actual,predicted)
1-mis_error5

# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=i) 
  
  predicted=predict(model6,test_set) # Predicting the test set
  
  mis_error6[i]=length(actual[actual!=predicted])/length(actual)
  sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(loan_status)~.,data=train_set,ntree=20)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
    formula = as.factor(loan_status) ~ .,
    data = train_set,
    nbagg = i,   
    coob = FALSE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  predicted=predict(model7,test_set) # Predicting the test set
  
  mis_error7[i]=length(actual[actual!=predicted])/length(actual)
  sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(loan_status) ~ .,
  data = train_set,
  nbagg = 20,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3

# SMOTE------------------------------------------------------------------
table(data1$loan_status)
library(smotefamily)
A=SMOTE(data1[,-6],data1$loan_status)
B=A$data
head(B)
table(B$class)
View(B)

# Breaking the dataset in train and test set-------------------------
train_size <- round(0.8 * nrow(B))
train_indices = createDataPartition(B$class,p=0.8)$Resample1

# Create training and testing sets--------
train_set = B[train_indices, ]
test_set = B[-train_indices, ]

nrow(train_set)
nrow(test_set)

table(train_set$class)

as.numeric(train_set$class)
actual=as.numeric(test_set$class)
test_set=test_set[,-16]

model1 <- flac(as.numeric(train_set$class)~.,data = train_set, family = "binomial")
summary(model1)

prob1=predict(model1,newdata = test_set,type = "response")
length(prob1)

mis_error1=array(0)
sen1=array(0) #TPR
prec1=array(0)
fpr1=array(0)

for(i in 1:999)
{
  predicted=ifelse(prob1>cutoff[i],1,0)
  mis_error1[i]=length(actual[actual!=predicted])/length(actual)
  sen1[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec1[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
  fpr1[i]=length(actual[actual==0&predicted==1])/length(actual[actual==0])
}

error1=data.frame(cutoff,mis_error1)
ggplot(error1, aes(x = cutoff, y = mis_error1)) +
  geom_point() +
  labs(title = "Misclassification error for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Misclassification Error")


sensitivity1=data.frame(cutoff,sen1)
ggplot(sensitivity1, aes(x = cutoff, y = sen1)) +
  geom_point() +
  labs(title = "Sensitivity for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Sensitivity")

precision1=data.frame(cutoff,prec1)
ggplot(precision1, aes(x = cutoff, y = prec1)) +
  geom_point() +
  labs(title = "Precision for Different Choices of Cutoffs", 
       subtitle = "Logistic Regression Model",
       x = "cutoff", y = "Precision")

roc1=data.frame(sen1,fpr1)
ggplot(roc1, aes(x = fpr1, y = sen1)) +
  geom_point() +
  labs(title = "ROC Curve for Logistic Regression Model", 
       x = "FPR", y = "TPR")

predicted=ifelse(prob1>0.5,1,0)
mis_error1=length(actual[actual!=predicted])/length(actual)
sen1=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec1=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr1=length(actual[actual==0&predicted==1])/length(actual[actual==0])

roc1=roc(actual,prob1)
auc(roc1)
1-mis_error1
# Decision Tree----------------------------------------------
model5=tree(as.factor(train_set$class)~.,data=train_set)
summary(model5)

plot(model5)  # plotting the decision tree
text(model5,pretty = 0)  # adding labels

predicted=predict(model5,test_set,type="class")  # predicting test set
predicted

mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5
# Random Forest---------------------------------------------
mis_error6=array(0)
sen6=array(0)
prec6=array(0)

for(i in 1:20)
{
  model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=i) 
  
  predicted=predict(model6,test_set) # Predicting the test set
  
  mis_error6[i]=length(actual[actual!=predicted])/length(actual)
  sen6[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec6[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=randomForest(as.factor(train_set$class)~.,data=train_set,ntree=17)
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

# Bagging----------------------------------------------------
mis_error7=array(0)
sen7=array(0)
prec7=array(0)

for(i in 1:20)
{
  model7 <- bagging(
    formula = as.factor(train_set$class) ~ .,
    data = train_set,
    nbagg = i,   
    coob = FALSE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  predicted=predict(model7,test_set) # Predicting the test set
  
  mis_error7[i]=length(actual[actual!=predicted])/length(actual)
  sen7[i]=length(actual[actual==1&predicted==1])/length(actual[actual==1])
  prec7[i]=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
}

model6=bagging(
  formula = as.factor(train_set$class) ~ .,
  data = train_set,
  nbagg = 17,   
  coob = FALSE,
  control = rpart.control(minsplit = 2, cp = 0))
model6
predicted=predict(model6,test_set) # Predicting the test set
table(actual,predicted)
mis_error5=length(actual[actual!=predicted])/length(actual)
sen5=length(actual[actual==1&predicted==1])/length(actual[actual==1])
prec5=length(actual[actual==1&predicted==1])/length(predicted[predicted==1])
fpr5=length(actual[actual==0&predicted==1])/length(actual[actual==0])
1-mis_error5

df=data.frame(n=1:20,mis_error6,mis_error7)
g1=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=mis_error6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=mis_error7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Misclassification Error in Test Set") + 
  ggtitle("Misclassification Error vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df1=data.frame(n=1:20,sen6,sen7)
g2=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=sen6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=sen7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Sensitivity in Test Set") + 
  ggtitle("Sensitivity vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))

df2=data.frame(n=1:20,prec6,prec7)
g3=ggplot(df, aes(x = n)) + 
  geom_line(data=df, aes(x=n, y=prec6, color="Random Forest")) + 
  geom_line(data=df, aes(x=n, y=prec7, color="Bagging")) + 
  xlab("Number of Trees") + 
  ylab("Precision in Test Set") + 
  ggtitle("Precision vs Number of Trees") + 
  scale_color_manual(name="", values=c("Random Forest"="blue", "Bagging"="red"))
g1
g2+g3
