# importing necessary libraries 
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from data_helper import *
import datetime

# Data processing
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = flattenData(trainData), flattenData(validData), flattenData(testData)
#trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

# training a linear SVM classifier 

print(datetime.datetime.now())
svm_model_linear = SVC(kernel='linear', C=1).fit(trainData, trainTarget)
print(datetime.datetime.now())

svm_predictions = svm_model_linear.predict(testData)

# model accuracy for X_test   
accuracy = svm_model_linear.score(trainData, trainTarget)
print(accuracy)

# creating a confusion matrix 
cm = confusion_matrix(testTarget, svm_predictions)
print(cm)
