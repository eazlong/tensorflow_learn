from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels= ['A','A','B','B']
	return group, labels

def classify0( inX, dataSet, labels, k ):
	dataSetSize = dataSet.shape[0]
	diffMat = tile( inX, ( dataSetSize, 1 ) ) - dataSet
	print( tile( inX, ( dataSetSize, 1 ) ) )
	print( diffMat )
	sqDiff = diffMat**2
	print( sqDiff )
	sqSum = sqDiff.sum( axis=1 )
	print( sqSum )
	distances = sqSum**0.5
	sortDisInd = distances.argsort()
	classCount = {}
	for i in range(k):
		voteILabels = labels[sortDisInd[i]]
		classCount[voteILabels] = classCount.get( voteILabels, 0 ) + 1
	sortedClassCount = sorted( classCount.items(), key=operator.itemgetter(1), reverse=True )
	return sortedClassCount[0][0]

def file2mat( filename ):
	f = file.open( filename )
	arrayOnLines = f.readlines()
	num = len( arrayOnLines )
	returnMat = zeros((num, 3))
	classLabelVectors = []
	index = 0
	for line in arrayOnLines:
		line = line.strip()
		listFromLine = line.split( '\t' )
		returnMat[index,:]=listFromLine[0:3]
		classLabelVectors.append( int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVectors

def autoNormal( dataSet ):
	mins = dataSet.min(0)
	maxs = dataSet.max(0)
	ranges = maxs-mins;
	normalDataSet = zeros( shape( dataSet ) )
	m = dataSet.shape[0]
	normalDataSet = dataSet-tile( mins, ( m, 1 ) )
	normalDataSet = normalDataSet/tile( ranges, ( m, 1 ) )
	return normalDataSet, ranges, mins

data, labels = createDataSet()
print( classify0( [1,0], data, labels, 2 ) )

