from math import log
import operator

def calcShannonEnt( dataSet ):
	numEntries = len( dataSet )
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float( labelCounts[key])/numEntries
		shannonEnt -= prob*log( prob, 2 )
	return shannonEnt

def splitDataSet( dataSet, axis, value ):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			tmpDataSet = featVec[:axis]
			tmpDataSet.extend( featVec[axis+1:] )
			retDataSet.append( tmpDataSet )
	return retDataSet

def chooseBestFeatureToSpit( dataSet ):
	numFeatures = len( dataSet[0] ) - 1
	baseEnt = calcShannonEnt( dataSet )
	bestInfoGain = 0.0
	bestFeatureIndex = -1
	for i in range( numFeatures ):
		featureList = [ example[i] for example in dataSet ]
		print( featureList )
		uniqueVals = set( featureList )
		print( uniqueVals )
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet( dataSet, i, value )
			prob = len( subDataSet )/float(len( dataSet ) )
			newEntropy += prob * calcShannonEnt( subDataSet )
		infoGain = baseEnt - newEntropy
		if ( infoGain > bestInfoGain ):
			bestInfoGain = infoGain
			bestFeatureIndex = i
	return bestFeatureIndex

def createDataSet():
	group = [[1,1,'yes'],[1,1,'no'],[0,1,'no'],[1,0,'yes'],[1,0,'yes']]
	labels= ['no surfuring','fellipers']
	return group, labels

def majorityCount( classList ):
	countDict = {}
	for classV in classList:
		if classV not in countDict.keys():
			countDict[classV] = 0;
		countDict[classV] += 1
	sortedCount = sorted( countDict.items(), key=operator.itemgetter(1), reverse=True )
	return sortedCount[0][0]

def buildTree( dataSet, labels ):
	classList = [example[-1] for example in dataSet]
	if ( classList.count( classList[0]) == len( classList ) ):
		return classList[0]d
	if ( len(dataSet[0]) == 1 ):
		return majorityCount( classList )
	bestFeatureIndex = chooseBestFeatureToSpit( dataSet )
	bestFeatLabel = labels[bestFeatureIndex]
	myTree = {bestFeatLabel:{}}
	featValues = [example[bestFeatureIndex] for example in dataSet]
	uniqueVals = set( featValues )
	for value in featValues:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = buildTree( splitDataSet( dataSet, bestFeatureIndex, value ), subLabels )
	return myTree

def classify( inputTree, featLabel, testVec ):
	firstStr = list(myTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabel.index( firstStr )
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type( secondDict[key] ).__name__ == 'dict' :
				classLabel = classify( secondDict[key], featLabel, testVec )
			else:
				classLabel = secondDict[key]
	return classLabel


dataSet,labels = createDataSet()
classLabel = [ example[-1] for example in dataSet ]
print( classLabel )
print( classLabel[0] )
print( classLabel.count( classLabel[0] ) )
print( chooseBestFeatureToSpit( dataSet ) )
myTree = buildTree( dataSet, labels )
print( myTree )
print( classify( myTree, labels, [1,1] ) )
print( classify( myTree, labels, [0,1] ) )
print( classify( myTree, labels, [1,0] ) )
print( classify( myTree, labels, [0,0] ) )

