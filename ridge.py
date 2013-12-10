#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from setup import *

def main():

	initData()

	X = vectorizeText(get('train')['tweet'], 10000)
	# X = load('X_100000')
	# testX = vectorizeText(data_test['tweet'])

	print 'Vectorized'

	trainData = X[:50000]
	testData = X[50000:]
	trainLabels = get('complete_list')[:50000]
	testLabels = get('complete_list')[50000:]

	print testData.shape
	print testLabels.shape

	clf = []
	predictions = []
	for i in xrange(24):
		print 'Starting %d...' % i
		clf.append(trainRidgeRegressor(trainData, trainLabels[:,i]))
		predictions.append(clf[i].predict(testData))

	predictions = removeOutOfBounds(predictions)
	predictions = np.transpose(predictions)
	# predictions = removeOutOfBounds(predictions)
	# writePredictions(predictions)
	# col = '%f,'*23 + '%f'
	# np.savetxt('out_prediction1.csv', predictions, col, delimiter=',')

	print('S: Train Error', trainError(predictions[:,0:5], testLabels[:,0:5]))
	print('W: Train Error', trainError(predictions[:,5:10], testLabels[:,5:10]))
	print('K: Train Error', trainError(predictions[:,10:], testLabels[:,10:]))
	print('All: Train Error', trainError(predictions, testLabels))

if __name__ == '__main__':
	main()