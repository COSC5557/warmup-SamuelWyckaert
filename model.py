import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


# Read data from csv file
with open('data/winequality-red.csv', 'r') as f:
    reader = csv.reader(f)
    redWine = list(reader)
    redWine = redWine[1:]
with open('data/winequality-white.csv', 'r') as f:
    reader = csv.reader(f)
    whiteWine = list(reader)
    whiteWine = whiteWine[1:]


#combine the two datasets and specify the type of wine
for i in range(len(redWine)):
    redWine[i].append('red')
for i in range(len(whiteWine)):
    whiteWine[i].append('white')
allWine = redWine + whiteWine
trainingSet, testingSet = train_test_split(allWine, test_size=0.20, random_state=1)

trainingResult = []
testResult = []

# Convert data to float
for i in range(len(trainingSet)):
    trainingSet[i] = trainingSet[i][0].split(';')
    for j in range(len(trainingSet[i])):
        trainingSet[i][j] = float(trainingSet[i][j])
    trainingResult.append(trainingSet[i][-1])
    trainingSet[i].remove(trainingSet[i][-1])


#use a dummy classifier to compare the results
dummy = DummyClassifier(strategy="uniform")
dummy.fit(trainingSet, trainingResult)
print("Baseline (uniforme) :", dummy.score(trainingSet, trainingResult))


# Train the models
knnModel = KNeighborsClassifier(n_neighbors=1)
knnModel.fit(trainingSet, trainingResult)
regressionModel = LinearRegression().fit(trainingSet, trainingResult)

# Predict
for i in range(len(testingSet)):
    testingSet[i] = testingSet[i][0].split(';')
    for j in range(len(testingSet[i])):
        testingSet[i][j] = float(testingSet[i][j])
    testResult.append(testingSet[i][-1])
    testingSet[i].remove(testingSet[i][-1])


# get the accuracy of the model
print("accuracy of the KNN model with K = 1 :", knnModel.score(testingSet, testResult))
print("accuracy of the linear regression model :", regressionModel.score(testingSet, testResult))


"""
#train with different values of k and stock the accuracy in an array
accuracy = [0]
for i in range(1, 250):
    knnModel = KNeighborsClassifier(n_neighbors=i)
    knnModel.fit(trainingSet, trainingResult)
    print("accuracy of the KNN model with k = ", i, ":", knnModel.score(testingSet, testResult))
    accuracy.append(knnModel.score(testingSet, testResult))

#plot the accuracy
plt.plot(accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
"""


