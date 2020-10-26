##################################################
# ML Workshop for SER 2020
# Rachel Sippy
# October 26, 2020
##################################################

#file location
setwd("~/SISA/")

#load the example dataset
sisa<-readRDS("~/SISA/Example1.RData")

#the caret library houses all of the machine learning machinery
library(caret)

##################################################
# step 1: prepare your data for use in caret
##################################################

#this creates dummy variables automatically
dsisa3<-dummyVars(~.,data=sisa,fullRank=TRUE)
dsisa<-data.frame(predict(dsisa3,newdata=sisa))

#variables with zero or near zero variance are not useful to us
#check for nearzero vars
nzv <- nearZeroVar(dsisa, saveMetrics= TRUE)
nzv

#very highly correlated variables are also not useful to us
descrCor <-  cor(dsisa)
highCorr <- sum(abs(descrCor[upper.tri(dsisa)]) > .999)
highCorr

#linear combinations of variables that equal other 
#variables are also not useful
comboInfo <- findLinearCombos(dsisa)
comboInfo

#re-create factor outcome
dsisa$Hospitalized.yes<-factor(dsisa$Hospitalized.yes,labels=c("no","yes"))
colnames(dsisa)[30]<-"Hospitalized"

##################################################
# step 2: create test and training sets
##################################################

#create training set
#you always have to set a seed if you want to get the same split
set.seed(1984)
sis.part <- createDataPartition(y = dsisa$Hospitalized,
                                ## the outcome data need to be specified
                                p = .85,
                                ## The percentage of data in the training set
                                list = FALSE)

sis.train<-dsisa[sis.part,]#sis.part is the list of selected training obs
sis.test<-dsisa[-sis.part,]

##################################################
# step 3: set fit controls to determine how best
#         algorithm is chosen, some algorithms 
#         need additional controls (tuning grid)
##################################################

fitControl <- trainControl(
    method = "repeatedcv",## repeated k-fold CV
    number = 10,## define k
    repeats = 10,## repeated ten times
    classProbs=TRUE, ## this enables us to calculate ROC
    summaryFunction=twoClassSummary)

##################################################
# step 4: train all your algorithms
##################################################

####################logistic regression

set.seed(1984)#ensure that the same resamples are used
lr.sis <- train(Hospitalized ~ ., data = sis.train, 
                method = "glm",
                family="binomial",
                trControl=fitControl,
                metric="ROC")

#mean ROC, sens, spec from CV training
#this is your modeling object, created from all the training data
lr.sis 

#influence of variables
lrImp<-varImp(lr.sis,scale=FALSE)
plot(lrImp,main="Logistic Regression")

#####################bagged trees

set.seed(1984)
bag.sis <- train(Hospitalized ~ ., data = sis.train, 
                 method = "treebag",
                 trControl=fitControl,
                 verbose = FALSE,
                 metric="ROC")

#mean ROC, sens, spec from CV training
bag.sis

#influence of variables
bagImp<-varImp(bag.sis,scale=FALSE)
plot(bagImp,main="Bagged Trees")

#using kappa or accuracy metric
fitControl2 <- trainControl(
    method = "repeatedcv",## repeated k-fold CV
    number = 10,## define k
    repeats = 10) ## repeated ten times

set.seed(1984)
bag.sis2 <- train(Hospitalized ~ ., data = sis.train, 
                 method = "treebag",
                 trControl=fitControl2,
                 verbose = FALSE)

#mean kappa, accuracy from CV training
bag.sis2

#hist of performance metrics in CV &CV repeats
resampleHist(bag.sis, main="Bagged Trees",col=1)
resampleHist(bag.sis2, main="Bagged Trees",col=1)

#####################k-nearest neighbors, parameter: k

#tuning grid for parameter fit
knnGrid <-  expand.grid(k = c(1:15))#test 1 to 15 neighbors

set.seed(1984)
knn.sis <- train(Hospitalized ~ ., data = sis.train, 
                 method = "knn",
                 trControl=fitControl,
                 tuneGrid=knnGrid,
                 metric="ROC")

#mean accuracy and kappa from CV training
knn.sis

#influence of variables
knnImp<-varImp(knn.sis,scale=FALSE)
plot(knnImp,main="K-Nearest Neighbors")

#hist of ROC, sens, spec in CV &CV repeats
resampleHist(knn.sis, main="K-Nearest Neighbors")

#plots for models with parameters
trellis.par.set(caretTheme())
plot(knn.sis,main="K-Nearest Neighbors")


####################elastic net, parameters: lambda, fraction

#tuning grid for parameters 
enetGrid <- expand.grid(alpha = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
                        lambda = c(0,0.001,0.01,0.1,0.25,0.5,1,2,2.5,3))

set.seed(1984)
enet.sis <- train(Hospitalized ~ ., data = sis.train, 
                  method = "glmnet",
                  trControl=fitControl,
                  tuneGrid=enetGrid,
                  metric="ROC")

#mean ROC, sens, spec from CV training
enet.sis

#hist of ROC,sens, sepc in CV &CV repeats
resampleHist(enet.sis, main="Elastic Net Regression")

#influence of variables
enetImp<-varImp(enet.sis,scale=FALSE)
plot(enetImp,main="Elastic Net Regression")

#plots for models with parameters
trellis.par.set(caretTheme())
plot(enet.sis,main="Elastic Net Regression")
plot(enet.sis, metric="Kappa",main="Elastic Net Regression")
plot(enet.sis,plotType='level',main="Elastic Net Regression")

####################generalized boosting models, 
####################parameters: interaction.depth, n.trees,
####################shrinkage, n.minobsinnode

#tuning grid for parameters 
gbmGrid <-  expand.grid(interaction.depth = c(1,5,9), 
                        n.trees=(1:5)*30, 
                        shrinkage=0.1,
                        n.minobsinnode=10)#make n.mino<10 for small trainings

set.seed(1984)
gbm.sis <- train(Hospitalized~.,data=sis.train,
                 method = "gbm",
                 trControl=fitControl,
                 tuneGrid=gbmGrid,
                 verbose=FALSE,
                 metric="ROC")

#mean ROC, sens, spec from CV training
gbm.sis

#influence of variables
gbmImp<-varImp(gbm.sis,scale=FALSE)
plot(gbmImp,main="Generalized Boosting Model")

#######################################group activity
#set.seed(1984)
#ba.sis <- train(Hospitalized ~ ., data = sis.train, 
#                method = "bayesglm",
#                trControl=fitControl,
#                metric="ROC")

#mean ROC, sens, spec from CV training
#ba.sis

#set.seed(1984)
#svm.sis <- train(Hospitalized ~ ., data = sis.train, 
#                method = "svmRadialSigma",
#                trControl=fitControl,
#                metric="ROC")

#mean ROC, sens, spec from CV training
svm.sis

####################random forest, parameter: mtry

#tuning grid for parameter 
rfGrid <-  expand.grid(mtry = c(1:12))

set.seed(1984)
rf.sis <- train(Hospitalized ~ ., data = sis.train, 
                method = "rf",
                trControl=fitControl,
                verbose = FALSE,
                tuneGrid=rfGrid,
                importance=TRUE,
                metric="ROC")

#mean ROC, sens, spec from CV training
rf.sis

#influence of variables
rfImp<-varImp(rf.sis,scale=FALSE)
plot(rfImp,main="Random Forest")

#####################neural networks, parameters: size, decay

#neural networks has tuning grid to fit other parameters 
nnetGrid <- expand.grid(size = c(1,3,5,10), #test these values for size
                        decay = c(0,0.1,0.25,0.5)) #test these for decay

set.seed(1984)#ensure that the same resamples are used
nnet.sis <- train(Hospitalized ~ ., data = sis.train, 
                  method = "nnet",
                  tuneGrid=nnetGrid,
                  verbose=FALSE,
                  trControl=fitControl,
                  metric="ROC")

#mean ROC, sens, spec from CV training
nnet.sis

#influence of variables
nnetImp<-varImp(nnet.sis,scale=FALSE)
plot(nnetImp,main="Neural Network")

#hist of performance metrics in CV &CV repeats
resampleHist(nnet.sis, main="Neural Network",col=1)

#plots for models with parameters
trellis.par.set(caretTheme())
plot(nnet.sis,main="Neural Network")
plot(nnet.sis, metric="ROC",main="Neural Network")
plot(nnet.sis,plotType='level',main="Neural Network")

##################################################
# step 5: use your trained algorithms (saved as 
#         xxx.sis) to make predictions on your
#         test set
##################################################

#make predictions and save
bagPred<-predict(bag.sis,newdata=sis.test)
knnPred<-predict(knn.sis,newdata=sis.test)
rfPred<-predict(rf.sis,newdata=sis.test)
gbmPred<-predict(gbm.sis,newdata=sis.test)
enetPred<-predict(enet.sis,newdata=sis.test)
nnetPred<-predict(nnet.sis,newdata=sis.test)
lrPred<-predict(lr.sis,newdata=sis.test)

##################################################
# step 6: assess the quality of the final  
#         prediction using your test metric
##################################################

#calculate accuracy and kappa for predictions
#versus observations in test set
bag<-postResample(pred=bagPred,obs=sis.test$Hospitalized)
knn<-postResample(pred=knnPred,obs=sis.test$Hospitalized)
final<-data.frame(bag=bag,knn=knn)
final$rf<-postResample(pred=rfPred,obs=sis.test$Hospitalized)
final$gbm<-postResample(pred=gbmPred,obs=sis.test$Hospitalized)
final$enet<-postResample(pred=enetPred,obs=sis.test$Hospitalized)
final$nnet<-postResample(pred=nnetPred,obs=sis.test$Hospitalized)
final$lr<-postResample(pred=lrPred,obs=sis.test$Hospitalized)

#calculate AUC-ROC for predictions versus observations 
#in test set
library(pROC)

sis.test$predn<-predict(nnet.sis,sis.test)
nnord<-factor(sis.test$predn,ordered=TRUE,levels=c("yes","no"))
obj.roc <- roc(sis.test$Hospitalized,nnord )

sis.test$predb<-predict(bag.sis,sis.test)
bord<-factor(sis.test$predb,ordered=TRUE,levels=c("yes","no"))
b.roc <- roc(sis.test$Hospitalized,bord )

sis.test$predk<-predict(knn.sis,sis.test)
kord<-factor(sis.test$predk,ordered=TRUE,levels=c("yes","no"))
k.roc <- roc(sis.test$Hospitalized,kord )

sis.test$predr<-predict(rf.sis,sis.test)
rord<-factor(sis.test$predr,ordered=TRUE,levels=c("yes","no"))
r.roc <- roc(sis.test$Hospitalized,rord )

sis.test$prede<-predict(enet.sis,sis.test)
eord<-factor(sis.test$prede,ordered=TRUE,levels=c("yes","no"))
e.roc <- roc(sis.test$Hospitalized,eord )

sis.test$predg<-predict(gbm.sis,sis.test)
gord<-factor(sis.test$predg,ordered=TRUE,levels=c("yes","no"))
g.roc <- roc(sis.test$Hospitalized,gord )

sis.test$predl<-predict(lr.sis,sis.test)
lord<-factor(sis.test$predl,ordered=TRUE,levels=c("yes","no"))
l.roc <- roc(sis.test$Hospitalized,lord )
