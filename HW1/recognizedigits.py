import numpy as np
import random as rd
import pylab as py

#	Convert matrices into arrays
def matrix2Arr(matrix):
	flat = [item for sublist in matrix for item in sublist]
	return np.array(flat)	

#	Convert matrices into lists
def matrix2List(matrix):
	flat = [item for sublist in matrix for item in sublist]
	return flat

#	display patterns in a more eye-friendly matter
def display(pattern):
    py.imshow(pattern.reshape((16,10)), cmap=py.cm.binary, interpolation='nearest')
    py.show()

def patternToVector(x1, x2, x3, x4, x5):
	#Convert pattern-matrix to list
	x1List = matrix2List(x1)
	x2List = matrix2List(x2)
	x3List = matrix2List(x3)
	x4List = matrix2List(x4)
	x5List = matrix2List(x5)

	#Add all patterns to one list
	patternsList = []
	patternsList.append(x1List)
	patternsList.append(x2List)
	patternsList.append(x3List)
	patternsList.append(x4List)
	patternsList.append(x5List)

	#Convert list to array
	patternsArr = np.array(patternsList)

	return patternsArr

#	Train Weight-matrix with array of all patterns (as vectors)
def trainWeights(patterns):
	_,col = np.shape(patterns)
	#_,col = np.shape(patterns)			#	Gets dimensions for weight matrix	
	Weights = np.zeros((col,col))	
	for i in patterns:					
	    Weights += np.outer(i,i)		#	W-matrix(new) = W-vector(old) plus pattern-vector * pattern-vector^T
	Weights[np.diag_indices(col)] = 0	#	Removes Identity-matrix	
	return Weights/col					#   Weight-matrix * 1/p, (divided by number of patterns) change to number of neurons 1/N

#	Asynchronus testing of patterns
def asyncTest(W, pattern):	
	sgn = lambda x: -1 if x<0 else +1
	steadyState = True
	newPattern = np.copy(pattern)		
	i = 0
	epoch = 0

	while(not(steadyState and i ==159)):#	Iterates over neurons until steady state is achieved 
		for i in range(0,160):			#	Typewriter index
			if (i == 0):
				steadyState = True
			newState = sgn(np.dot(newPattern, W[:,i]))
			if (newState != newPattern[i]):
				steadyState = False

			newPattern[i] = newState
		epoch += 1					
	return newPattern

#	Training patterns
x1=[ 
	[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] 
	];

x2=[ 
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
	[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ];

x3=[ 
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
	[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
	[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
	[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
	[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] 
	];

x4=[ 
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
	[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] 
	];

x5=[ 
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
	[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
	[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] 
	];

#	Question patterns

	#1
question1 = [
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1], 
			[1, -1, -1, 1, 1, -1, 1, -1, -1, -1]
			]
	#2			
question2 = [
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
			[1, 1, 1, 1, 1, -1, -1, -1, 1, 1], 
			[1, 1, 1, 1, 1, -1, -1, -1, 1, 1], 
			[1, 1, 1, 1, 1, -1, -1, -1, 1, 1], 
			[1, 1, 1, 1, 1, -1, -1, -1, 1, 1], 
			[1, 1, 1, 1, 1, -1, -1, -1, 1, 1], 
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
			[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], 
			[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1]
			]
	#3			
question3 = [
			[-1, 1, -1, -1, -1, -1, 1, -1, 1, 1], 
			[1, -1, -1, -1, 1, -1, -1, 1, 1, -1], 
			[-1, 1, 1, -1, 1, -1, -1, 1, -1, 1], 
			[-1, 1, 1, -1, 1, -1, -1, -1, -1, -1], 
			[1, 1, 1, 1, -1, 1, -1, -1, -1, 1], 
			[-1, 1, 1, 1, -1, -1, -1, 1, -1, 1], 
			[1, -1, 1, 1, -1, 1, -1, 1, -1, -1], 
			[-1, -1, 1, 1, -1, -1, -1, 1, -1, 1], 
			[-1, -1, -1, 1, -1, -1, -1, 1, -1, -1], 
			[1, -1, -1, 1, 1, 1, -1, 1, 1, -1], 
			[1, 1, 1, 1, 1, 1, -1, -1, 1, -1], 
			[1, -1, 1, 1, -1, 1, -1, 1, -1, 1], 
			[1, 1, 1, 1, -1, 1, 1, 1, 1, 1], 
			[-1, -1, -1, -1, -1, 1, -1, 1, -1, 1], 
			[1, -1, 1, -1, 1, 1, 1, -1, -1, 1], 
			[-1, 1, 1, 1, -1, -1, -1, -1, 1, 1]
			]

# Convert training patterns to vector array
patternsArr = patternToVector(x1, x2, x3, x4, x5)

# Train weights with vector array
weights = trainWeights(patternsArr)

#	Convert question-patterns to vector
question1Arr = matrix2Arr(question1)
question2Arr = matrix2Arr(question2)
question3Arr = matrix2Arr(question3)

#	Test convergence of questions with weight-matrix
question1Fix = asyncTest(weights, question1Arr)
question2Fix = asyncTest(weights, question2Arr)
question3Fix = asyncTest(weights, question3Arr)

print("Question 1 fixed: ")
print(repr(question1Fix.reshape(16,10)))

print("Question 2 fixed: ")
print(repr(question2Fix.reshape(16,10)))

print("Question 3 fixed: ")
print(repr(question3Fix.reshape(16,10)))

#	Display patterns before and after convergece
display(question1Arr)

display(question1Fix)

display(question2Arr)

display(question2Fix)

display(question3Arr)

display(question3Fix)