library(ipft)
library(readr)
library(caret)
library(e1071)
library(C50)
library(scatterplot3d)
library(plotly)

#-----------------> Set Local Variables --------------------------------------------------

threshold = -10 #  The decibel threshold that you want to filter (i.e. remove all values less than -90 (-90 to -100) and replace with 100)


#------------------> ENABLE LOCAL FILE ----------------------------------------
trainingData <- read_csv("C:/Users/josep/Downloads/UJIndoorLoc/UJIndoorLoc/trainingData.csv",
                         col_types = cols(BUILDINGID = col_character(),
                                          FLOOR = col_character(), PHONEID = col_character(),
                                          RELATIVEPOSITION = col_character(),
                                          USERID = col_character()))

#------------------> ENABLE GOOGLE DRIVE FILE ----------------------------------------
# trainingData <- read_csv("https://drive.google.com/uc?id=14oWD_ryEpUDb221zXwtN9ZFwkP7pvB2J", 
#                          col_types = cols(BUILDINGID = col_character(), 
#                                           FLOOR = col_character(), PHONEID = col_character(), 
#                                           RELATIVEPOSITION = col_character(), 
#                                           USERID = col_character()))

#----------- > Preprocess Data --------------------
trainingData$FLOOR = as.factor(trainingData$FLOOR)
trainingData$LATITUDE = as.numeric(trainingData$LATITUDE)
trainingData$BUILDINGID =as.factor(trainingData$BUILDINGID)
trainingData$SPACEID=as.factor(trainingData$SPACEID)
trainingData$RELATIVEPOSITION=as.factor(trainingData$RELATIVEPOSITION)
trainingData$USERID=as.factor(trainingData$USERID)
trainingData$PHONEID=as.factor(trainingData$PHONEID)

#----------- > Set Sample Size --------------------
sampleIndex = sample(1:nrow(trainingData), 100)#<--------Size of your sample...smaller= faster but less accurate.

#----------- > create Sample Dataframe called microTraining --------------------
microTraining = trainingData[sampleIndex,]

#----------- > Remove all Rows with NA  ---------------------------------------------------------------------------
microTraining = na.omit((microTraining))

#----------- > Remove all columns with only 100's --------------------
shrunk = microTraining[, !apply(microTraining == 100, 2, all)] #<----- Removes all solid 100 Columns

shrunk$LONGITUDE = shrunk$LONGITUDE*-1  #<---------------Change Longitude to a positive value (to keep it from being filtered)

#----------- > Replace all values less than the rheshold DB with 100 --------------------
for(i in 1:ncol(shrunk)) #<----------  Removes all values for decibel threshold
{
  shrunk[which(shrunk[,i] < threshold ), i] = 100
}

shrunk$LONGITUDE = shrunk$LONGITUDE*-1 #<---------------Change Longitude back to a negative value

# #----------- > Remove all columns with only 100's --------------------
shrunk = shrunk[, !apply(shrunk == 100, 2, all)]


#------------> Enable if there are issues with removing 100's column or filtering DB values---------
#shrunk = microTraining# <------Disabled...Jumper bypass to avoid removing all "100" values


#-------------> Create Training/Testing Partitions -------------------------------------------------
inTraining <- createDataPartition(shrunk$BUILDINGID, p = .75, list = FALSE)
training <- na.omit(shrunk[inTraining,])
testing <- shrunk[-inTraining,]

#--------------> Another Pre-Processing step because for some reason R just wants to make these both the wrong data type------
training$LATITUDE = as.numeric(training$LATITUDE)
training$BUILDINGID = as.factor(training$BUILDINGID)




#------ Legend for Column Locations-----##
#520 = Last WAP       ncol(trainingData)-9
#521 = Long           ncol(trainingData)-8
#522 = Lat            ncol(trainingData)-7
#523 = Floor          ncol(trainingData)-6
#524 = Building ID    ncol(trainingData)-5
#525 = SPace ID       ncol(trainingData)-4
#526 = Rel Pos        ncol(trainingData)-3
#527 = UserID         ncol(trainingData)-2
#528 = PhoneID        ncol(trainingData)-1
#529 = DateTime       ncol(trainingData)-0





############################# MODEL CASCADE CHOSEN #############################################
###########------------> Model Competition for predicting Building ID using WAP -------------------

#---------------------------------<<<<<< Classification PROBLEM >>>>>>>>>>>>> ----------------------------------------------------

training = na.omit(training)
#-------------------------------------------------------------------> FUNCTIONAL
modelknnBLD <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))],
                     method = "knn", 
                     # trControl = trainControl(), 
                     preProcess = c("center","scale")) 

# k  Accuracy   Kappa    
# 5  0.9752859  0.9615057
# 7  0.9777332  0.9653213
# 9  0.9761350  0.9627317

#-------------------------------------------------------------------> FUNCTIONAL
modelsvmBLD <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))],
                     method = "svmLinear2", 
                     #trControl = fitControl, 
                     preProcess = c("center","scale")) 

#cost  Accuracy   Kappa    
#0.25  0.9950433  0.9922809
#0.50  0.9950433  0.9922809
#1.00  0.9950433  0.9922809

#-------------------------------------------------------------------> FUNCTIONAL
training = na.omit(training)
modelGBMbld = train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))],
                    method = "gbm", 
                    preProcess = c("center","scale") )

# interaction.depth  n.trees  Accuracy   Kappa    
# 1                   50      0.9787249  0.9663245
# 1                  100      0.9808209  0.9696561
# 1                  150      0.9819962  0.9714611

#-------------------------------------------------------------------> FUNCTIONAL
modelRFbld <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))], 
                    method = "rf",  
                    ntree = 1)

#-------------------------------------------------------------------> FUNCTIONAL
modelNBbld <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))], 
                    method = "nb",  
                    preProcess = c("center","scale"))


#-------------------------------------------------------------------> FUNCTIONAL
modelC50bld <- train(BUILDINGID ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5))], 
                     method = "C5.0",  
                     preProcess = c("center","scale"))

#------ Legend for Column Locations-----##
#520 = Last WAP       ncol(trainingData)-9
#521 = Long           ncol(trainingData)-8
#522 = Lat            ncol(trainingData)-7
#523 = Floor          ncol(trainingData)-6
#524 = Building ID    ncol(trainingData)-5
#525 = SPace ID       ncol(trainingData)-4
#526 = Rel Pos        ncol(trainingData)-3
#527 = UserID         ncol(trainingData)-2
#528 = PhoneID        ncol(trainingData)-1
#529 = DateTime       ncol(trainingData)-0


###########------------> STEP 2-A Model Competition for predicting Lat Using WAP and Building ID-------------------

#---------------------------------<<<<<< REGRESSION PROBLEM >>>>>>>>>>>>> ----------------------------------------------------

training = na.omit(training)
#-------------------------------------------------------------------> FUNCTIONAL
modelknnLAT <- train(LATITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-7))],
                     method = "knn", 
                     
                     preProcess = c("center","scale")) 

# k  RMSE      Rsquared   MAE     
# 5  15.26252  0.9493048   9.54672
# 7  15.68233  0.9470450  10.06308
# 9  15.84787  0.9466506  10.42121

#-------------------------------------------------------------------> FUNCTIONAL
modelsvmLAT <- train(LATITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-7))],
                     method = "svmLinear2", 
                     #trControl = fitControl, 
                     preProcess = c("center","scale")) 

#-------------------------------------------------------------------> FUNCTIONAL
modelGBMlat = train(LATITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-7))],
                    method = "gbm", 
                    preProcess = c("center","scale") )

#-------------------------------------------------------------------> NOT FUNCTIONAL
modelRFlat <- train(LATITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-7))], 
                    method = "rf",  
                    ntree = 1)
modelRFlat
#-------------------------------------------------------------------> wrong model type for regression
# modelNBlat <- train(LATITUDE ~ ., data = training[,c(1:522,524)], 
#                     method = "nb",  
#                     preProcess = c("center","scale"))
# modelNBlat
#-------------------------------------------------------------------> wrong model type for regression
# modelC50lat <- train(LATITUDE ~ ., data = training[,c(1:522, 524)], 
#                      method = "C5.0",  
#                      preProcess = c("center","scale"))
# modelC50lat    

#------ Legend for Column Locations-----##
#520 = Last WAP       ncol(trainingData)-9
#521 = Long           ncol(trainingData)-8
#522 = Lat            ncol(trainingData)-7
#523 = Floor          ncol(trainingData)-6
#524 = Building ID    ncol(trainingData)-5
#525 = SPace ID       ncol(trainingData)-4
#526 = Rel Pos        ncol(trainingData)-3
#527 = UserID         ncol(trainingData)-2
#528 = PhoneID        ncol(trainingData)-1
#529 = DateTime       ncol(trainingData)-0

###########------------>  STEP 2-B Model Competition for predicting Long Using WAP and Buildingr-------------------

#---------------------------------<<<<<< REGRESSION PROBLEM >>>>>>>>>>>>> ----------------------------------------------------

training = na.omit(training)
#-------------------------------------------------------------------> FUNCTIONAL
modelknnLONG <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8))],
                      method = "knn", 
                      #trControl = fitControl, 
                      preProcess = c("center","scale")) 

# k  RMSE      Rsquared   MAE     
# 5  31.36249  0.9369185  16.52921
# 7  31.29304  0.9400378  17.91355
# 9  32.05852  0.9402061  19.11787

#-------------------------------------------------------------------> FUNCTIONAL
modelsvmLONG <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8))],
                      method = "svmLinear2", 
                      #trControl = fitControl, 
                      preProcess = c("center","scale")) 

# cost  RMSE      Rsquared   MAE     
# 0.25  31.64321  0.9336384  22.91115
# 0.50  33.46023  0.9263760  23.73234
# 1.00  36.32606  0.9143231  25.30051

#-------------------------------------------------------------------> FUNCTIONAL
modelGBMlong = train(LONGITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8))],
                     method = "gbm", 
                     preProcess = c("center","scale") )

#-------------------------------------------------------------------> FUNCTIONAL
modelRFlong <- train(LONGITUDE ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8))], 
                     method = "rf",  
                     ntree = 1)

#-------------------------------------------------------------------> Not a Regression Model
# modelNBlong <- train(LONGITUDE ~ ., data = training[,c(1:522,524)], 
#                      method = "nb",  
#                      preProcess = c("center","scale"))
# 
# modelNBlong

# #-------------------------------------------------------------------> wrong model type for regression
# modelC50long <- train(LONGITUDE ~ ., data = training[,c(1:522, 524)],
#                      method = "C5.0",
#                      preProcess = c("center","scale"))
# modelC50long

#------ Legend for Column Locations-----##
#520 = Last WAP       ncol(trainingData)-9
#521 = Long           ncol(trainingData)-8
#522 = Lat            ncol(trainingData)-7
#523 = Floor          ncol(trainingData)-6
#524 = Building ID    ncol(trainingData)-5
#525 = SPace ID       ncol(trainingData)-4
#526 = Rel Pos        ncol(trainingData)-3
#527 = UserID         ncol(trainingData)-2
#528 = PhoneID        ncol(trainingData)-1
#529 = DateTime       ncol(trainingData)-0

###########------------> Step 3 Model Competition for predicting floor Using WAP and Building and lat longr-------------------

#---------------------------------<<<<<< Classification PROBLEM >>>>>>>>>>>>> ----------------------------------------------------
training = na.omit(training)
#-------------------------------------------------------------> FUNCTIONAL
modelknnFloor <- train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))],
                       method = "knn", 
                       #trControl = fitControl, 
                       preProcess = c("center","scale")) 

# k  Accuracy   Kappa    
# 5  0.5819402  0.4567261
# 7  0.5921799  0.4695678
# 9  0.6016781  0.4816753

#-------------------------------------------------------------> FUNCTIONAL
modelsvmFloor <- train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))],
                       method = "svmLinear2", 
                       #trControl = fitControl, 
                       preProcess = c("center","scale")) 

# cost  Accuracy   Kappa    
# 0.25  0.7875237  0.7232133
# 0.50  0.7875237  0.7232133
# 1.00  0.7875237  0.7232133

#-------------------------------------------------------------> FUNCTIONAL
modelGBMfloor = train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))],
                      method = "gbm", 
                      preProcess = c("center","scale") )





#-------------------------------------------------------------> FUNCTIONAL
modelRFfloor <- train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))], 
                      method = "rf",  
                      ntree = 1)



#-------------------------------------------------------------> FUNCTIONAL
modelNBfloor <- train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))],
                      method = "nb",  
                      preProcess = c("center","scale"))

#-------------------------------------------------------------------> FUNCTIONAL
modelC50floor <- train(FLOOR ~ ., data = training[, c(1:(ncol(shrunk)-9), (ncol(shrunk)-5), (ncol(shrunk)-8), (ncol(shrunk)-7), (ncol(shrunk)-6))],
                       method = "C5.0",
                       preProcess = c("center","scale"))


#summaries

modelknnBLD   #<----- Functional (@100: 0.8651192; @1000: 0.9809717)  (-97: 0.9860925)  (-95: 0.9904596)  (-90: 0.9579615)  (-85: 0.9470924)  (-70:  0.8497217)
modelsvmBLD   #<----- Functional (@100: 0.995997; @1000:  0.9933827)  (-97: 0.9913591)  (-95: 0.9916903)  (-90: 0.9848484)  (-85: 0.9785025)  (-70:  0.8879426)
modelGBMbld   #<----- Functional (@100: 0.7542142; @1000: 0.9933554) >(-97: 0.9968525)< (-95: 0.9949117)  (-90: 0.9900118)  (-85: 0.9818704)  (-70:  0.8599952)
modelRFbld    #<----- Functional (@100: 0.8692397; @1000: 0.9678213)  (-97: 0.9672617)  (-95: 0.9665623)  (-90: 0.9628810)  (-85: 0.9507846)  (-70:  0.8793040)
modelNBbld    #<----- Functional (@100: 0.5198998; @1000: 0.5074266)  (-97: 0.4884003)  (-95: 0.488366)   (-90: 0.499807 )  (-85: 0.4886081)  (-70:  0.4591607)
modelC50bld   #<----- Functional (@100: 0.8058532; @1000: 0.9880289)  (-97: 0.9880697)  (-95: 0.9848035)  (-90: 0.9801068)  (-85: 0.9750852)  (-70:  0.7282144)

modelknnLAT   #<----- Functional (@100: 0.6962331; @1000: 0.9458061)  (-97: 0.9513668)  (-95: 0.9512664)  (-90: 0.9376725)  (-85: 0.9306593)  (-70:  0.8913413)
modelsvmLAT   #<----- Functional (@100: 0.7925616; @1000: 0.8911071)  (-97: 0.8811075)  (-95: 0.8804652)  (-90: 0.8832494)  (-85: 0.8936691)  (-70:  0.8592778)
modelGBMlat   #<----- Functional (@100: 0.8274798; @1000: 0.9516668) >(-97: 0.9535214)< (-95: 0.9490763)  (-90: 0.9489082)  (-85: 0.9479064)  (-70:  0.8836043)
modelRFlat    #<----- Functional (@100: 0.6980582; @1000: 0.8965931)  (-97: 0.9175727)  (-95: 0.9104273)  (-90: 18.25691)   (-85: 0.9287079)  (-70:  0.9258618)
#modelNBlat   #<----- Classification Model, Not For Regession
#modelC50lat  #<----- Classification Model, Not For Regession

modelknnLONG   #<----- Functional (@100: 0.8281039; @1000: 0.9663286) >(-97: 0.9664600)< (-95: 0.9620523)  (-90: 0.9380920)  (-85: 0.9397278)  (-70:  0.9425770)
modelsvmLONG   #<----- Functional (@100: 0.9059247; @1000: 0.929937)   (-97: 0.9214888)  (-95: 0.9144890)  (-90: 0.9170852)  (-85: 0.9228856)  (-70:  0.9053729)
modelGBMlong   #<----- Functional (@100: 0.8460866; @1000: 0.9823931)  (-97: 0.9818009)  (-95: 0.9813404)  (-90: 0.9791754)  (-85: 0.9797464)  (-70:  0.9515945)
modelRFlong    #<----- Functional (@100: 0.8671330; @1000: 0.9716193)  (-97: 0.9693190)  (-95: 0.9693508)  (-90: 0.9666420)  (-85: 0.9739650)  (-70:  0.9689030)
#modelNBlong   #<----- Classification Model, Not For Regession
#modelC50long  #<----- Classification Model, Not For Regession

modelknnFloor  #<----- Functional (@100: 0.2537349; @1000: 0.7331208)  (-97: 0.7168041)  (-95: 0.7451603)  (-90: 0.7294661)  (-85: 0.7562804)  (-70:  0.6398552)
modelsvmFloor  #<----- Functional (@100: 0.5393549; @1000: 0.8196711)  (-97: 0.8121590)  (-95: 0.8267071)  (-90: 0.8196711)  (-85: 0.8387511)  (-70:  0.7020588)
modelGBMfloor  #<----- Functional (@100: 0.3838227; @1000: 0.8203565) >(-97: 0.8363415)< (-95: 0.8225903)  (-90: 0.8144038)  (-85: 0.8379875)  (-70:  0.7160980)
modelRFfloor   #<----- Functional (@100: 0.3897196; @1000: 0.6424264)  (-97: 0.6517895)  (-95: 0.6427010)  (-90: 0.6770332)  (-85: 0.6936111)  (-70:  0.6747763)
modelNBfloor   #<----- Functional (@100: 0.2576539; @1000: 0.05808455) (-97: 0.0718803)  (-95: 0.0882408)  (-90: 0.0512595)  (-85: 0.0617978)  (-70:  0.0450079)
modelC50floor  #<----- Functional (@100: 0.4180020; @1000: 0.7735700)  (-97: 0.7973760)  (-95: 0.7778795)  (-90: 0.7879656)  (-85: 0.8180120)  (-70:  0.7362754)

################################# End Model Cascade ########################################################

#--------> Select Building Model ----------------

#buildingModel = modelknnBLD 
# buildingModel = modelsvmBLD 
# buildingModel = modelGBMbld 
# buildingModel = modelRFbld 
# buildingModel = modelNBbld 
# buildingModel = modelC50bld

#--------> Select Lat Model -------------------

#latitudeModel = modelknnLAT 
# latitudeModel = modelsvmLAT
# latitudeModel = modelGBMlat 
# latitudeModel = modelRFlat  

#--------> Select Long Model -------------------

#longitudeModel = modelknnLONG  
# longitudeModel = modelsvmLONG  
# longitudeModel = modelGBMlong  
# longitudeModel = modelRFlong  

#--------> Select Floor Model -------------------

#floorModel = modelknnFloor 
# floorModel = modelsvmFloor
# floorModel = modelGBMfloor
# floorModel = modelRFfloor 
# floorModel = modelNBfloor 
# floorModel = modelC50floor 

################ Begin Predictions ######################
#--------> Building Predictions ------------------
buildingPredictionknn <- predict(modelknnBLD, testing)
buildingPredictionsvm <- predict(modelsvmBLD, testing)
buildingPredictiongbm <- predict(modelGBMbld, testing)
buildingPredictionrf <- predict(modelRFbld, testing)
buildingPredictionnb <- predict(modelNBbld, testing)
buildingPredictionc50 <- predict(modelC50bld, testing)

#----------> Prep work----------------------

evaluation = as.data.frame(testing$LATITUDE)
evaluation$LONGITUDE = as.data.frame(testing$LONGITUDE)
LATpredictions = as.data.frame(testing$LATITUDE)
LONGpredictions = as.data.frame(testing$LONGITUDE)

#--------> Lattitude Predictions ------------------
latitudePredictionknn <- predict(modelknnLAT, testing)
LATpredictions$knn = testing$LATITUDE - (predict(modelknnLAT, testing))
latitudePredictionsvm <- predict(modelsvmLAT, testing)
LATpredictions$svm = testing$LATITUDE - (predict(modelsvmLAT, testing))
latitudePredictiongbm <- predict(modelGBMlat, testing)
LATpredictions$gbm = testing$LATITUDE - (predict(modelGBMlat, testing))
latitudePredictionrf <- predict(modelRFlat, testing)
LATpredictions$rf = testing$LATITUDE - (predict(modelRFlat, testing))


#--------> Longitude Predictions ------------------
longitudePredictionknn <- predict(modelknnLONG, testing)
LONGpredictions$knn = testing$LONGITUDE - (predict(modelknnLONG, testing))
longitudePredictionsvm <- predict(modelsvmLONG, testing)
LONGpredictions$svm = testing$LONGITUDE - (predict(modelsvmLONG, testing))
longitudePredictiongbm <- predict(modelGBMlong, testing)
LONGpredictions$gbm = testing$LONGITUDE - (predict(modelGBMlong, testing))
longitudePredictionrf <- predict(modelRFlong, testing)
LONGpredictions$rf = testing$LONGITUDE - (predict(modelRFlong, testing))

#---------->Calculate LAT/LONG Error (CEP)----------

evaluation$knnError = as.data.frame( sqrt( ( LONGpredictions$knn )^2 + ( LATpredictions$knn )^2 ))
evaluation$svmError = as.data.frame( sqrt( ( LONGpredictions$svm )^2 + ( LATpredictions$svm )^2 ))
evaluation$gbmError = as.data.frame( sqrt( ( LONGpredictions$gbm )^2 + ( LATpredictions$gbm )^2 ))
evaluation$rfError = as.data.frame( sqrt( ( LONGpredictions$rf )^2 + ( LATpredictions$rf )^2 ))

colnames(evaluation) =c("Real LAT", "Real Long", "KNN Error", "SVM Error", "GBM Error", "RF Error")

summary(evaluation)

#------------> Prep Data for Plotting------------------------------------
knnPred$long = as.data.frame(longitudePredictionknn)
knnPred$lat = as.data.frame(latitudePredictionknn)
svmPred$long = as.data.frame(longitudePredictionsvm)
svmPred$lat = as.data.frame(latitudePredictionsvm)
gbmPred$long = as.data.frame(longitudePredictiongbm)
gbmPred$lat = as.data.frame(latitudePredictiongbm)
rfPred$long = as.data.frame(longitudePredictionrf)
rfPred$lat = as.data.frame(latitudePredictionrf)


#--------> KNN Versus Testing Data----------------------------------------
knn = ggplot() + 
  geom_point(data = testing, aes(x = LATITUDE, y = LONGITUDE), color = "black") +
  geom_point(data = knnPred, aes(x = lat, y = long), color = "blue") +
  
  xlab('Latitude') +
  ylab('Longitude')+
  labs(title = "KNN Versus Testing Data")


#--------> SVM Versus Testing Data----------------------------------------
svm = ggplot() + 
  geom_point(data = testing, aes(x = LATITUDE, y = LONGITUDE), color = "black") +
  geom_point(data = svmPred, aes(x = lat, y = long), color = "red") +
  
  xlab('Latitude') +
  ylab('Longitude')+
  labs(title = "SVM Versus Testing Data")


#--------> GBM Versus Testing Data----------------------------------------
gbm = ggplot() + 
  geom_point(data = testing, aes(x = LATITUDE, y = LONGITUDE), color = "black") +
  geom_point(data = gbmPred, aes(x = lat, y = long), color = "yellow") +
  
  xlab('Latitude') +
  ylab('Longitude')+
  labs(title = "GBM Versus Testing Data")


#--------> RF Versus Testing Data----------------------------------------
rf = ggplot() + 
  geom_point(data = testing, aes(x = LATITUDE, y = LONGITUDE), color = "black") +
  geom_point(data = rfPred, aes(x = lat, y = long), color = "green") +
  
  xlab('Latitude') +
  ylab('Longitude')+
  labs(title = "RF Versus Testing Data")

#----------> Grid Arrange to plot all together--------------

grid.arrange(knn, svm, gbm, rf ,ncol = 2, nrow = 2)
#-------------------------------------------------------


plot(longitudePredictionknn, latitudePredictionknn)
plot(longitudePredictionsvm, latitudePredictionsvm)
plot(longitudePredictiongbm, latitudePredictiongbm)
plot(longitudePredictionrf, latitudePredictionrf)
plot(testing$LONGITUDE, testing$LATITUDE)

#--------> Floor Predictions ------------------
floorPredictionknn <- predict(modelknnFloor, testing)
floorPredictionsvm <- predict(modelsvmFloor, testing)
floorPredictiongbm <- predict(modelGBMfloor, testing)
floorPredictionrf <- predict(modelRFfloor, testing)
floorPredictionnb <- predict(modelNBfloor, testing)
floorPredictionc50 <- predict(modelC50floor, testing)

################ Check Predictions ######################
#--------> Building Predictions ------------------
postResample(buildingPredictionknn, testing$BUILDINGID)     #  <<<<<<<< BUILDING PREDICTION @7000 SAMPLE SIZE
# Accuracy     Kappa 
# 0.9988558 0.9982004  
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.9959839 0.9936530 

#------> -97--------
# Accuracy     Kappa 
# 0.9880000 0.9814306
#------> -95--------
# Accuracy    Kappa 
# 1        1 
#
#------> -85--------
# Accuracy     Kappa 
# 0.9440000 0.9124716
#
#------> -70--------
# Accuracy     Kappa 
# 0.3654618 0.0116317 

postResample(buildingPredictionsvm, testing$BUILDINGID)
# Accuracy     Kappa   
# 0.9977117 0.9964042  
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.9959839 0.9936530 

#------> -97--------
# Accuracy     Kappa 
# 0.9960000 0.9938103 
#------> -95--------
# Accuracy    Kappa 
# 1        1 
#
#------> -85--------
# Accuracy     Kappa 
# 0.9840000 0.9747691 
#
#------> -70--------
# Accuracy       Kappa 
# 0.34939759 -0.02168077 

postResample(buildingPredictiongbm, testing$BUILDINGID)
# Accuracy     Kappa
# 0.9977117 0.9964042  
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.9959839 0.9936651

#------> -97--------
#Accuracy     Kappa 
# 0.9960000 0.9938103 
#------> -95--------
# Accuracy     Kappa 
# 0.9959839 0.9937505
#
#------> -85--------
# Accuracy     Kappa 
# 0.9920000 0.9873658 
#
#------> -70--------
# Accuracy      Kappa 
# 0.3493976 -0.0230541 

postResample(buildingPredictionrf, testing$BUILDINGID)
# Accuracy     Kappa 
# 0.9948513 0.9919032 
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.9759036 0.9622832 

#------> -97--------
#Accuracy     Kappa 
# 0.9840000 0.9751337 
#------> -95--------

# Accuracy     Kappa 
# 0.9397590 0.9046635 
#
#------> -85--------
# Accuracy     Kappa 
# 0.9800000 0.9684104
#
#------> -70--------
# Accuracy       Kappa 
# 0.34136546 -0.03578947 

postResample(buildingPredictionnb, testing$BUILDINGID)
# Accuracy     Kappa 
# 0.5903890 0.2525467
#
#   All Run at 1000
#------> NO Filter--
# Accuracy      Kappa 
# 0.50200803 0.05681818 

#------> -97--------
# Accuracy     Kappa 
# 0.5160000 0.1345769 
#------> -95--------
# Accuracy     Kappa 
# 0.5301205 0.1523959
#
#------> -85--------
# Accuracy    Kappa 
# 0.484    0.000 
#
#------> -70--------
# Accuracy     Kappa 
# 0.4698795 0.0000000 

postResample(buildingPredictionc50, testing$BUILDINGID)
# Accuracy     Kappa 
# 0.9959954 0.9937037 
#
#   All Run at 1000
#------> NO Filter--
# Accuracy    Kappa 
# 1        1 

#------> -97--------
#Accuracy    Kappa 
# 0.992000 0.987621
#------> -95--------
# Accuracy     Kappa 
# 0.9959839 0.9937505 
#
#------> -85--------
# Accuracy     Kappa 
# 0.9880000 0.9810227 
#
#------> -70--------
# Accuracy       Kappa 
# 0.35341365 -0.01800406 

#--------> Latitude Predictions ------------------
postResample(latitudePredictionknn, testing$LATITUDE)     #  <<<<<<<< LATITUDE PREDICTION @7000 SAMPLE SIZE
# RMSE     Rsquared          MAE 
# 4.864761e+06 9.765534e-01 4.864761e+06 
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared        MAE 
# 11.3453131  0.9751248  7.5286581 

#------> -97--------
#      RMSE   Rsquared        MAE 
# 11.3630755  0.9707145  7.4162835 
#------> -95--------
# RMSE   Rsquared        MAE 
# 13.4802230  0.9581161  8.5796412
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 13.6306366  0.9606967  8.2919167 
#
#------> -70--------
# RMSE     Rsquared          MAE 
# NA 0.0004246061           NA 

postResample(latitudePredictionsvm, testing$LATITUDE)
# RMSE     Rsquared          MAE 
# 4.864761e+06 9.534191e-01 4.864761e+06  
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared       MAE 
# 21.023031  0.912598 14.727654 

#------> -97--------
# #      RMSE   Rsquared        MAE 
# 21.2888647  0.9032295 14.7529557 
#------> -95--------
# RMSE   Rsquared        MAE 
# 19.8446061  0.9072688 15.0742088 
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 19.5223283  0.9177679 14.3122410
#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.002677987          NA 

postResample(latitudePredictiongbm, testing$LATITUDE)
# RMSE     Rsquared          MAE 
# 4.864761e+06 9.625011e-01 4.864761e+06 
#
#   All Run at 1000
#------> NO Filter--
# RMSE   Rsquared        MAE 
# 14.8299507  0.9557192 10.9465641 

#------> -97--------
# #      RMSE   Rsquared        MAE 
# 12.7974661  0.9631687  9.5588016 
#------> -95--------
# RMSE   Rsquared        MAE 
# 13.7725878  0.9565717 10.1913488 
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 13.8350113  0.9593867 10.7963613 
#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.001899954          NA 

postResample(latitudePredictionrf, testing$LATITUDE)
# RMSE     Rsquared          MAE 
# 4.864762e+06 9.677327e-01 4.864762e+06
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared        MAE 
# 15.5006565  0.9523416  9.8408364 

#------> -97--------
# #      RMSE   Rsquared        MAE 
# 17.1964242  0.9337499 10.9841413 
#------> -95--------
# RMSE  Rsquared       MAE 
# 17.108582  0.931599 10.345642 
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 16.5105219  0.9410122 10.6424436
#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.004279214          NA 


#--------> Longitude Predictions ------------------
postResample(longitudePredictionknn, testing$LONGITUDE)
# RMSE     Rsquared          MAE 
# 217.06579547   0.00118301 182.42523417
#
#   All Run at 1000
#------> NO Filter--
# RMSE         Rsquared        MAE 
# 15.8945614  0.9850427 10.0949169 

#------> -97--------
# #     RMSE   Rsquared        MAE 
# 16.8359135  0.9812371  9.8153939 
#------> -95--------
# RMSE   Rsquared        MAE 
# 17.9618631  0.9788721 11.1962223 
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 26.5136477  0.9527653 12.4695204 
#
#------> -70--------
# RMSE   Rsquared        MAE 
# NA 0.00313634         NA 

postResample(longitudePredictionsvm, testing$LONGITUDE)
# RMSE     Rsquared          MAE 
# 2.357096e+06 9.827547e-01 2.357096e+06
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared        MAE 
# 32.0886039  0.9360195 22.8315692

#------> -97--------
# #     RMSE   Rsquared        MAE 
# 32.3982021  0.9318557 23.0513261
#------> -95--------
# RMSE   Rsquared        MAE 
# 30.8048545  0.9341899 21.3851279
#
#------> -85--------
######################################
###################################

#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.003911454          NA 


postResample(longitudePredictiongbm, testing$LONGITUDE)
# RMSE    Rsquared         MAE 
# 153.1557519   0.4528222 118.5794034
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared        MAE 
# 13.7121826  0.9884004 10.8827698 

#------> -97--------
# #     RMSE   Rsquared        MAE 
# 15.2018451  0.9848109 11.6683774 
#------> -95--------
# RMSE  Rsquared       MAE 
# 19.272402  0.974235 14.103311
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 15.8634590  0.9830842 11.9623017 
#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.003871796          NA 

postResample(longitudePredictionrf, testing$LONGITUDE)     #  <<<<<<<< LONGITUDE PREDICTION @7000 SAMPLE SIZE
# RMSE   Rsquared        MAE 
# 43.4791681  0.9114154 31.5759186 
#
#   All Run at 1000
#------> NO Filter--
# RMSE        Rsquared        MAE 
# 18.0881269  0.9794754 11.2249717 

#------> -97--------
# #      RMSE   Rsquared        MAE 
# 19.6525871  0.9748671 12.5378862 
#------> -95--------                   <<<<<<<< KNN FLOOR PREDICTION @1000 SAMPLE SIZE [82.8%]
# RMSE  Rsquared       MAE 
# 16.506927  0.981164 11.084944
#
#------> -85--------
# RMSE   Rsquared        MAE 
# 17.0599867  0.9802569 11.2123240 
#
#------> -70--------
# RMSE    Rsquared         MAE 
# NA 0.003684053          NA 


#--------> Floor Predictions ------------------
postResample(floorPredictionknn, testing$FLOOR)
# Accuracy      Kappa 
# 0.24199085 0.01112986  
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.7590361 0.6817078 

#------> -97--------        <<<<<<<< KNN FLOOR PREDICTION @1000 SAMPLE SIZE [82.8%]
# # Accuracy     Kappa 
# 0.8280000 0.7773866 
#------> -95--------
# Accuracy     Kappa 
# 0.7710843 0.6989884
#
#------> -85--------
# Accuracy     Kappa 
# 0.8200000 0.7656787
#
#------> -70--------
# Accuracy      Kappa 
# 0.23293173 0.00912557 

postResample(floorPredictionsvm, testing$FLOOR)
# Accuracy      Kappa 
# 0.05949657 0.00000000
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.8674699 0.8256414

#------> -97 -------
# # Accuracy     Kappa 
# 0.8520000 0.8083259 
#------> -95--------
# Accuracy     Kappa 
# 0.8714859 0.8316145 
#
#------> -85--------        <<<<<<<< SVM FLOOR PREDICTION @1000 SAMPLE SIZE [87.6%]
# Accuracy     Kappa 
# 0.8760000 0.8396375 
#
#------> -70--------
# Accuracy      Kappa 
# 0.24899598 0.03034152 

postResample(floorPredictiongbm, testing$FLOOR)
# Accuracy     Kappa 
# 0.9605263 0.9490509 
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.8795181 0.8413170

#------> -97--------
# # Accuracy     Kappa 
# 0.8960000 0.8656247
#------> -95--------
# Accuracy     Kappa 
# 0.8594378 0.8164954
#
#------> -85--------        <<<<<<<< GBM FLOOR PREDICTION @1000 SAMPLE SIZE [91.6%]
# Accuracy     Kappa 
# 0.9160000 0.8910675
#
#------> -70--------
# Accuracy       Kappa 
# 0.232931727 0.007056809 

postResample(floorPredictionrf, testing$FLOOR)
# Accuracy     Kappa 
# 0.8209382 0.7686184
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.6746988 0.5700215 

#------> -97--------
# #Accuracy     Kappa 
# 0.6880000 0.5967158
#------> -95--------
# Accuracy     Kappa 
# 0.6907631 0.5966806
#
#------> -85--------        <<<<<<<< RF FLOOR PREDICTION @1000 SAMPLE SIZE [73.6%]
# Accuracy     Kappa 
# 0.7360000 0.6585336
#
#------> -70--------
# Accuracy      Kappa 
# 0.2048193 -0.0292908 

postResample(floorPredictionnb, testing$FLOOR)
# Accuracy      Kappa 
# 0.08352403 0.02090291
#
#   All Run at 1000
#------> NO Filter--
# Accuracy      Kappa 
# 0.02811245 0.00000000 

#------> -97--------        <<<<<<<< NB FLOOR PREDICTION @1000 SAMPLE SIZE [7.2%]
# # Accuracy      Kappa 
# 0.07200000 0.01099838
#------> -95--------
# Accuracy       Kappa 
# 0.052208835 0.003493302 
#
#------> -85--------
# Accuracy    Kappa 
# 0.076    0.000 
#
#------> -70--------
# Accuracy      Kappa 
# 0.05220884 0.00000000 

postResample(floorPredictionc50, testing$FLOOR)     
# Accuracy     Kappa 
# 0.9736842 0.9660324 
#
#   All Run at 1000
#------> NO Filter--
# Accuracy     Kappa 
# 0.8433735 0.7944631 

#------> -97--------         <<<<<<<< C 5.0 FLOOR PREDICTION @1000 SAMPLE SIZE [84.8%]
# #Accuracy     Kappa 
# 0.8480000 0.8037879 
#------> -95--------
# Accuracy    Kappa 
# 0.811245 0.750825
#
#------> -85--------
# Accuracy     Kappa 
# 0.8640000 0.8241149
#
#------> -70--------
# Accuracy      Kappa 
# 0.25702811 0.04077213





#------------------> Making some Histograms of the WiFi Signal --------------------------------------
duration = training[,1:(ncol(shrunk)-9)]
histogram= filter(gather(duration,key,value),duration!=100)
hist(histogram$value)

building = as.numeric(shrunk$BUILDINGID)
hist(building)

floor = as.numeric(shrunk$FLOOR)
hist(floor)

spaceid = as.numeric(shrunk$SPACEID)
hist(spaceid)

#---------------> 3D Plot of Lat/Long/Floor ---------------
plotlyTrainingData <- plot_ly(type = "scatter3d" , mode = "markers" , x = training$LATITUDE, y = training$LONGITUDE, z = training$FLOOR, color = training$FLOOR, colors = c('#BF382A','green' ,'#0C4B8E')) %>%
  add_trace(type = "scatter3d" , mode = "markers" , x = training$LATITUDE, y = training$LONGITUDE, z = training$FLOOR, colors = "black") %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Latitude'),
                      yaxis = list(title = 'Longitude'),
                      zaxis = list(title = 'Floor')))

plotlyTrainingData
