data<- read.csv("C:\\Users\\HP\\OneDrive\\Desktop\\Msc sem 4 data\\Project\\Breast data\\new data.csv")
#View(data)
dim(data)
table(data$status)
data$X<- NULL
data$Status<-NULL
data <- as.data.frame(unclass(data),                     # Convert all columns to factor
                       stringsAsFactors = TRUE)

str(data)
#ALL LIBRARIES 
library(ggplot2)
library(farver)
library (e1071)
library(caret)
library(pROC)
library(randomForest)
library(class)
library(FNN)
#do alive=1,dead=0
#data$status<-as.factor(data$status)
#View(data)

########EDA


barplot(prop.table(table(data$status)),
        col=rainbow(2),
        main="class distribution")
#Age distributuion
hist(data$Age,probability = TRUE, main = "Age distribution of patients",col="green")

#Race distribution
plot(data$Race,col="red")
#count plot relationship between Race and Status
print(ggplot(data, aes(x=Race))+geom_bar())
#Plot a count plot to show the relationship between 'Marital Status' and 'Status'
print(ggplot(data, aes(x=Marital.Status))+geom_bar())

## plot to show relationship between Estrogen status and Status
print(ggplot(data, aes(x=Estrogen.Status))+geom_bar())
## plot to show relationship between progesterone status and Status
print(ggplot(data, aes(x=Progesterone.Status))+geom_bar())

# Create a plot to visualize the relationship between Tumor Size and Status
print(ggplot(data, aes(x=Tumor.Size))+geom_bar())
# Create a plot to visualize the relationship between N stage category and Status
print(ggplot(data, aes(x=N.Stage))+geom_bar())
# Create a plot to visualize the relationship between X6th.Stage  category and Status
print(ggplot(data, aes(x=X6th.Stage ))+geom_bar())
# Create a plot to visualize the relationship between A Stage  category and Status
print(ggplot(data, aes(x=A.Stage ))+geom_bar())
# Create a plot to visualize the relationship between grade and Status
print(ggplot(data, aes(x=Grade))+geom_bar())

#correlation b/w X6th.Stage and N.Stage
plot(data$X6th.Stage,data$N.Stage)


#TRAIN DATA

splits<-sample(2,nrow(data),replace=T,prob=c(0.7,0.3))
train<-data[splits==1,]
test<-data[splits==2,]

#..spliting the dataset in  training set and test set 100 times

train_test_splits=list()
for(i in 1:100)
{
 split<-createDataPartition(data$status,p=0.7,list=FALSE)
 train_data<-data[split,]
 test_data=data[-split,]
 
}



########logistic model
logistic<-glm(status ~Age+Race+Marital.Status+T.Stage + N.Stage +
                Grade+A.Stage +Tumor.Size+Estrogen.Status+Progesterone.Status+
                Regional.Node.Examined+Reginol.Node.Positive+Survival.Months,data = train,family = "binomial")    
summary(logistic)


#predict model
pmmodel= predict(logistic,test,type = "response")

#confution matrix in another way
tab= table(pmmodel>0.5,test$status)
tab

#accuracy of the matrix
sum(diag(tab))/sum(tab)*100

#ROC Curve
p <- predict(logistic, newdata=test)
roc(test$status,p,plot=TRUE)

#Area Under The ROC curve
auc(test$status,pmmodel)


#K-Fold cross Validation
cross_val=trainControl(method='cv',number=5)


#######Random Forest
model_accurecy=c()
test_model_accurecy=c()
for(i in 1:100)
{

 split<-createDataPartition(data$status,p=0.7,list=FALSE)
 train_data<-data[split,]
 test_data=data[-split,]

 RFM<-randomForest(factor(status)~Age+Race+Marital.Status+T.Stage + N.Stage +
                Grade+A.Stage +Tumor.Size+Estrogen.Status+Progesterone.Status+
                Regional.Node.Examined+Reginol.Node.Positive+Survival.Months,data= train_data,trControl=cross_val )
 model_accurecy[i]=sum(diag(RFM$confusion))/sum(RFM$confusion)*100
 
 pmmodel1= as.numeric(predict(RFM,newdata = test_data,type="response"))
 tab1= table(pmmodel1,test_data$status)
 test_model_accurecy[i]=sum(diag(tab1))/sum(tab1)*100


}
print(model_accurecy)#..model accurecy for train data

max(model_accurecy)
min(model_accurecy)

print(test_model_accurecy)#..model accurecy for test data
max(test_model_accurecy)
min(test_model_accurecy)



#..accurecy in training data
#sum(diag(RFM$confusion))/sum(RFM$confusion)*100

#Important feature of Random forest
#plot(RFM)
#varImpPlot(RFM,main="variable importance plot")



#predict model
#pmmodel1= as.numeric(predict(RFM,newdata = test,type="response"))


#confusion matrix
#tab1= table(pmmodel1,test$status)
#tab1
#accuracy on test data
#sum(diag(tab1))/sum(tab1)*100

# ROC curve
#p1 <- predict(RFM, newdata=test,type = "prob")


#roc(test$status,p1[,2],plot=TRUE)
#Area Under The ROC curve
#auc(test$status,pmmodel1)


#Roc comparison

roc(test$status,p,plot=TRUE,col="red",main="ROC Comparison")
par(new=TRUE)
roc(test$status,p1[,2],plot=TRUE,col="blue")


	#..svm

svmfit<-svm(factor(status)~Age+Race+Marital.Status+T.Stage + N.Stage +
                Grade+A.Stage +Tumor.Size+Estrogen.Status+Progesterone.Status+
                Regional.Node.Examined+Reginol.Node.Positive+Survival.Months,data= train,
                kernel="linear")
summary(svmfit)
plot(svmfit,data=train,kernel="linear")

# Predicting the Test set results 
pmmodel3 = predict(svmfit,test,type="decision") 

#confusion matrix
tab3= table(pmmodel3,test$status)
tab3

#accuracy
sum(diag(tab3))/sum(tab3)*100


###############.....KNN

# Execution of k-NN with k=1
knn1<- train(factor(status)~Age+Race+Marital.Status+T.Stage + N.Stage +
               Grade+A.Stage +Tumor.Size+Estrogen.Status+Progesterone.Status+
               Regional.Node.Examined+Reginol.Node.Positive+Survival.Months,data= train,
             method="knn",trControl=cross_val)
knn.k1<-knn1$bestTune
print(knn1)
plot(knn1)      #This chart shows the Elbow k = 9 with accuracy almost 90% for training dataset
                #Run prediction with test dataset and print out the confusion matrix:

# Predicting the Test set results 
pmmodel4 <- predict(knn1, newdata = test)

#confusion matrix
tab4= table(pmmodel4,test$status)
tab4

#accuracy
sum(diag(tab4))/sum(tab4)*100
#ROC Curve
p3 <- predict(knn1, newdata=test,prob=TRUE)
roc(test$status,as.numeric(p3),plot=TRUE)




# Fitting KNN Model to training dataset 
train2<-as.data.frame(sapply(train,as.numeric))
test2<-as.data.frame(sapply(test,as.numeric))

classifier_knn <- knn(train = train2[,-15], 
                      test = test2[,-15], 
                      cl = train2$status,
                      k = 9) 
qwe<-table(classifier_knn,test$status)
qwe
sum(diag(qwe))/sum(qwe)*100


###########################33333
test_model_accurecy=c()
for(i in 1:100)
{
  
  split<-createDataPartition(data$status,p=0.7,list=FALSE)
  train_data<-data[split,]
  test_data=data[-split,]
  
  knn1<- train(factor(status)~Age+Race+Marital.Status+T.Stage + N.Stage +
                 Grade+A.Stage +Tumor.Size+Estrogen.Status+Progesterone.Status+
                 Regional.Node.Examined+Reginol.Node.Positive+Survival.Months,data= train_data,
               method="knn",trControl=cross_val )
  
  pmmodel1= as.numeric(predict(knn1,newdata = test_data))
  tab1= table(pmmodel1,test_data$status)
  test_model_accurecy[i]=sum(diag(tab1))/sum(tab1)*100
  
}
#..Can't find model accurecy for train data
print(test_model_accurecy)#..model accurecy for test data
max(test_model_accurecy)
min(test_model_accurecy)
