from math import *
from numpy import *

def loadDataset():
	postingList = [['my','dog','is'], ['dog', 'pig', 'pig', 'fuck'], ['you', 'are', 'a', 'question'], ['you', 'are', 'not', 'a', 'question'] ]
	classVec = [0, 1, 2, 2]
	return postingList, classVec


def createVocabList( dataSet ):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet|set(document)
	return list(vocabSet)

def setOfWord2Vec( vocabSet, inputSet ):
	retVector = [0]*len(vocabSet)
	print( inputSet )
	for word in inputSet:
		if word in vocabSet:
			retVector[vocabSet.index(word)] += 1
		else:
			print( "word:%s not in vocate list" %word )
	return retVector

def trainNB0( trainMatix, trianCategory ):
	numTrainDocs = len( trainMatix )
	numWords = len( trainMatix[0] )
	cNum = max( trianCategory ) + 1
	pAbusive = [0]*cNum
	for i in range(len(trianCategory)):
		pAbusive[trianCategory[i]] += 1
	pAbusive = log(array(pAbusive)/array([float(numTrainDocs)]*cNum))
	pNumVec = array([[1]*numWords]*cNum)
	pDenon = [2.0]*cNum
	for i in range( numTrainDocs ):
		pNumVec[trianCategory[i]] += trainMatix[i]
		pDenon[trianCategory[i]] += sum( trainMatix[i] )
	pVect = log(pNumVec/array(pDenon).reshape(3,1))
	print( 'train NB:', numTrainDocs, numWords, pAbusive, pNumVec, pDenon, pVect )
	return pVect,pAbusive

def classifyNB( vec2Classify, pVec, pClass ):
	p = -float('inf')
	c = -1;
	for i in range(len(pVec)):
		p0 = sum( vec2Classify*pVec[i] ) + log(pClass[i])
		if p0 > p:
			p = p0
			c = i
	return c

def testingNB():
	dataSet, classVec = loadDataset()
	vocabList = createVocabList( dataSet )
	print( vocabList )
	trainMat = []
	for p in dataSet:
		trainMat.append( setOfWord2Vec( vocabList, p ) )
	print( "train mat ", trainMat )
	pV, pAb = trainNB0( trainMat, classVec )
	print( pV, pAb )

	mat = array( setOfWord2Vec( vocabList, [ 'pig', 'dog', 'like'] ) )
	print( "classified as:", classifyNB( mat, pV, pAb ) )

testingNB();
