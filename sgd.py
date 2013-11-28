#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from setup import *

def main():

	initData()

	X = vectorizeText(get('train')['tweet'], 10000)
	# X = load('X_10000')
	# testX = vectorizeText(data_test['tweet'])

	print 'Vectorized'

	trainData = X[:50000]
	testData = X[50000:]
	trainLabels = get('complete_list')[:50000]
	testLabels = get('complete_list')[50000:]

	clf = []
	predictions = []
	for i in xrange(24):
		print 'Starting %d...' % i
		clf.append(SGDRegressor(n_iter=100))
		clf[i].fit(trainData, trainLabels[:,i])
		predictions.append(clf[i].predict(testData))

	predictions = np.transpose(predictions)
	# predictions = removeOutOfBounds(predictions)
	# writePredictions(predictions)

	print('S: Train Error', trainError(predictions[:,0:5], testLabels[:,0:5]))
	print('W: Train Error', trainError(predictions[:,5:10], testLabels[:,5:10]))
	print('K: Train Error', trainError(predictions[:,10:], testLabels[:,10:]))
	print('All: Train Error', trainError(predictions, testLabels))

if __name__ == '__main__':
	main()