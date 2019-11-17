import numpy as np
import random as rd

def randomPatterns(x):
	randomPatterns = []
	values = [-1, 1]

	for _ in range(x):
		randomPattern = [0] * 120
		randomPattern = [values[rd.randint(0,1)] for i in randomPattern]

		randomPatterns.append(randomPattern)

	return np.array(randomPatterns)

def trainWeights(patterns):
    _,col = np.shape(patterns)		# 	Gets dimensions for weight matrix (and for division by number of neurons)
    Weights = np.zeros((col,col))		#	creates empty matrix
    for i in patterns:					
        Weights += np.outer(i,i)		#	W-new = W-old + pattern-vector * pattern-vector^T
    Weights[np.diag_indices(col)] = 0	#	Removes Identity-matrix	
    return Weights/col	

#	Same as above but with the diagonal patterns == 1
def trainWeightsWDiag(patterns):
    _,col = np.shape(patterns)		
    Weights = np.zeros((col,col))		
    for i in patterns:					
        Weights += np.outer(i,i)		
    return Weights/col	

def asyncTest(W, pattern):	
	sgn = lambda x: -1 if x<0 else +1
	errorCount = 0

	random = rd.randint(0,len(pattern)-1)

	newState = sgn(np.dot(pattern, W[:,random]))
	
	if (newState != pattern[random]):
		errorCount += 1	

	return errorCount


nErrors = []
errors = 0

x = [12, 24, 48, 70, 100, 120]

for p in x:
	#For bugtestingg
	print("p is now: ", end=' ')
	print(p)
	
	for _ in range(10**5):

		patterns = randomPatterns(p)

		weights = trainWeights(patterns)

		#weightsDiag = trainWeightsWDiag(patterns)					#train weights with diagonal values = 1

		random = rd.randint(0, (p-1))

		error = asyncTest(weights, patterns[random])

		errors += error

	errors = errors / (10**5)

	roundError = round(errors, 4)

	nErrors.append(roundError)

print(nErrors)


