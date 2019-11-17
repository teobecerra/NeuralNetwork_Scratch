import numpy as np
import random as rd
import math as ma

def initWeights():						
	return np.random.uniform(-0.2, 0.2,(1,4))

def calcOutput(vector, weights, threshold):
	return np.tanh(beta * (np.matmul(vector, np.transpose(weights)) - threshold))

def calcDeltaWeight(target, output, vector):
	return (learnRate * (target - output) * (1 - (output**2)) * vector * beta)

def calcDeltaThresh(target, output):
	return (-learnRate * ((target - output) * (1 - (output**2)) * beta))

sgn = lambda x: -1 if x<0 else +1

def checkIfLinear(output, target):
	for i in range(len(target)):
		if (not(output[i] == target[i])):
			return False
		return True

#	Initialize values
learnRate = 0.02
threshold = np.random.uniform(-1.0, 1.0)
beta = 0.5
weights = initWeights()

#	Input vectors
inputVectors = [
				[-1,-1,-1,-1],
				[1,-1,-1,-1],
				[-1,1,-1,-1],
				[-1,-1,1,-1],
				[-1,-1,-1,1],
				[1,1,-1,-1],
				[1,-1,1,-1],
				[1,-1,-1,1],
				[-1,1,1,-1],
				[-1,1,-1,1],
				[-1,-1,1,1],
				[1,1,1,-1],
				[1,1,-1,1],
				[1,-1,1,1],
				[-1,1,1,1],
				[1,1,1,1]
				]

#MY TARGETS
targetA = [1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
targetB = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1]
targetC = [-1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
targetD = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1]
targetE = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1]
targetF = [-1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1]

outputVector = [0] * 16

convergence = False

#	BUGTESTING

for j in range(10):

	print("trial number: ", j+1)

	for i in range(100000):

		mu = rd.randint(0,len(inputVectors)-1)

		output = calcOutput(inputVectors[mu], weights, threshold)

		outputVector[mu] = output
		
		deltaWeights = calcDeltaWeight(targetC[mu], output, inputVectors[mu])		#CHANGE FOR DIFFERENT TESTS

		deltaThresh = calcDeltaThresh(targetC[mu], output)								#CHANGE FOR DIFFERENT TESTS

		threshold += deltaThresh

		weights += deltaWeights

	for i in range(len(outputVector)):
		outputVector[i] = sgn(outputVector[i])

	if(outputVector == targetC):															#CHANGE FOR DIFFERENT TESTS
		convergence = True
		print()
		print("Convergence, have a cigarette!")
		break
	else:
		print()
		print("Still not converging")
		print()

if(convergence == False):
	print("No convergence my friend! Try another pattern")


print()
print("outputvector")
print(outputVector)
print()
print("targetC")																			#CHANGE FOR DIFFERENT TESTS
print(targetC)																				#CHANGE FOR DIFFERENT TESTS

print()
print("Threshhold is:", end=' ')
print(threshold)
print()

for w in weights:
	print("Weights are:", end=' ')
	print(w)