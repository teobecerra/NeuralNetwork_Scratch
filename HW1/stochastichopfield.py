import numpy as np
import random as rd
import math as ma

def randomPatterns(p):
	randomPatterns = []
	values = [-1, 1]

	for _ in range(p):
		randomPattern = [0] * 200
		randomPattern = [values[rd.randint(0,1)] for i in randomPattern]

		randomPatterns.append(randomPattern)

	return np.array(randomPatterns)

def trainWeights(patterns):
    _,col = np.shape(patterns)		# 	Gets dimensions for weight matrix (and for division by number of patterns)
    Weights = np.zeros((col,col))		#	creates empty matrix
    for i in patterns:					
        Weights += np.outer(i,i)		#	W-matrix(new) = W-vector(old) plus pattern-vector * pattern-vector^T
    Weights[np.diag_indices(col)] = 0	#	Removes Identity-matrix	
    return Weights/col	


def asyncTest(W, pattern):
	newPattern = np.copy(pattern)
	m = 0
	t = (2*(10**5))					

	for _ in range(t):					
		
		random = rd.randint(0,len(pattern)-1)

		newState = stochUpdate(np.dot(newPattern, W[:,random]))		
		
		newPattern[random] = newState

		m += calcM(newPattern,pattern) 

	return m / t 			
		

def calcM(patternWNoise, pattern):
	col = len(pattern)
	m = np.dot(patternWNoise,pattern)
	return  m / col 

def stochUpdate(b):	
	beta = 2
	gb = 1 / (1 + ma.exp(-2 * beta * b))

	state = np.random.choice([1, -1], size = None, p =[gb, 1-gb])

	return state


orderParameter = 0
nExperiments = 100					
p = 45								#7 for first question, 45 for second
									

for _ in range(nExperiments):			
	patterns = randomPatterns(p)

	weights = trainWeights(patterns)

	orderParameter += asyncTest(weights,patterns[0])
	
orderAvg = round((orderParameter / nExperiments), 3)

print("orderAvg is: ", end=' ')
print(orderAvg)



