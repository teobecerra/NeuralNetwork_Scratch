import numpy as np
import random as rd 
import csv

def initWeights(row, col):						
	weights = np.zeros((row,col))
	for r in range(len(weights)):
		for c in range(len(weights[r])):
			weights[r][c] = np.random.normal(0,1)
	return weights

def importData(filename):
	with open(filename) as csvfile:
		readCSV = csv.reader(csvfile)
		dataValues = []

		for row in csvfile:
			list = row.split(",")
			for i in range(len(list)):
				list[i] = float(list[i])
			dataValues.append(list)
	return dataValues

def feedLayer(layer, weights, threshold):
	return np.tanh(np.matmul(layer, weights) - threshold)

def calcError(layer, error, weight):
	return (1 - (layer * layer)) * np.matmul(error,np.transpose(weight))

def updateWeights(weight, layer,error, eta):
	return (weight + eta * (np.matmul(np.transpose(np.array(layer)[np.newaxis]),((error)[np.newaxis]))))

def updateThreshold(threshold, error, eta):
	return (threshold - (error * eta))

layerIn	= 	[0, 0]
layerV1 = 	[0, 0, 0, 0, 0]
layerV2 = 	[0, 0, 0, 0, 0, 0, 0, 0]
layerOut =	[0]

errorV1 = np.array([0, 0, 0, 0, 0])
errorV2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
errorOut = np.array([0])

threshV1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
threshV2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
threshOut = np.array([0.0])	

outputsTrain = [0] * 10000
outputsTest = [0] * 5000

eta = 0.02
diffTrain = 0
diffTest = 0

#	INITIALIZE WEIGHTS
w1 = initWeights(2,5) 											
w2 = initWeights(5,8)											
w3 = initWeights(8,1)											

#	IMPORT DATA VALUES

trainingValues = importData('training_set.csv')

testValues = importData('validation_set.csv')

#TRAINING PHASE

#	FORWARD PROPAGATION

for i in range(10000000):
	
	mu = rd.randint(0,9999)

	#INPUT LAYER
	for l in range(len(layerIn)):
		layerIn[l] = (trainingValues[mu][l])

	#HIDDEN LAYER 1
	layerV1 = feedLayer(layerIn, w1, threshV1)

	#HIDDEN LAYER 2
	layerV2 = feedLayer(layerV1, w2, threshV2)

	#OUTPUT LAYER 
	layerOut = feedLayer(layerV2, w3, threshOut)

#	BACKPROGAGATION

	#OUTPUT LAYER ERROR
	errorOut = (1 - (layerOut * layerOut)) * (trainingValues[mu][2] - layerOut)

	#HIDDEN LAYER 2 ERROR
	errorV2 = calcError(layerV2, errorOut, w3)
	
	#HIDDEN LAYER 1 ERROR
	errorV1 = calcError(layerV1, errorV2, w2)

#	UPDATE WEIGHTS

	w1 = updateWeights(w1, layerIn, errorV1, eta)

	w2 = updateWeights(w2, layerV1, errorV2, eta)

	w3 = updateWeights(w3, layerV2, errorOut, eta)

#	UPDATE THRESHOLDS
	
	threshV1 = updateThreshold(threshV1, errorV1, eta)

	threshV2 = updateThreshold(threshV2, errorV2, eta)

	threshOut = updateThreshold(threshOut, errorOut, eta)

#END OF TRAINING PHASE

print("training complete")

#TESTING PHASE

#	TEST TRAINING

for i in range(len(trainingValues)):
	
	mu = i

	#FORWARD PROPAGATION

	#INPUT LAYER
	for l in range(len(layerIn)):
		layerIn[l] = (trainingValues[mu][l])

	#HIDDEN LAYER 1
	layerV1 = feedLayer(layerIn, w1, threshV1)

	#HIDDEN LAYER 2
	layerV2 = feedLayer(layerV1, w2, threshV2)

	#OUTPUT LAYER 
	layerOut = feedLayer(layerV2, w3, threshOut)

	outputsTrain[i] = layerOut

for i in range(len(outputsTrain)):
	diffTrain += abs((np.sign(outputsTrain[i]) - trainingValues[i][2]))

cTrain = 1/20000 * diffTrain

#	TEST TESTVALUES

for i in range(len(testValues)):
	
	mu = i

	#FORWARD PROPAGATION

	#INPUT LAYER
	for l in range(len(layerIn)):
		layerIn[l] = (testValues[mu][l])

	#HIDDEN LAYER 1
	layerV1 = feedLayer(layerIn, w1, threshV1)

	#HIDDEN LAYER 2
	layerV2 = feedLayer(layerV1, w2, threshV2)

	#OUTPUT LAYER 
	layerOut = feedLayer(layerV2, w3, threshOut)

	outputsTest[i] = layerOut

for i in range(len(outputsTest)):
	diffTest += abs((np.sign(outputsTest[i]) - testValues[i][2]))

cTest = 1/10000 * diffTest

np.savetxt('w1.csv', np.transpose(w1), delimiter=',')
np.savetxt('w2.csv', np.transpose(w2), delimiter=',')
np.savetxt('w3.csv', np.transpose(w3), delimiter=',')

np.savetxt('t1.csv', threshV1, delimiter=',')
np.savetxt('t2.csv', threshV2, delimiter=',')
np.savetxt('t3.csv', threshOut, delimiter=',')

print("testing of trainingvalues complete")
print("cTrain: ", end=' ')
print(cTrain)

print("testing of trainingvalues complete")
print("cTest: ", end=' ')
print(cTest)

print()
print("		Thresholds after training")
print(threshV1)
print(threshV2)
print(threshOut)
print()
print("		Weights after training")
print("w1:", end=' ')
for w in w1:
	print(w, end=' ')
print()
print("w2:", end=' ')
for w in w2:
	print(w, end=' ')
print()
print("w3:", end=' ')
for w in w3:
	print(w, end=' ')
print()
print()




