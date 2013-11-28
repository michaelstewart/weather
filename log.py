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

	X = vectorizeText(get('tweet'), 6000)
	# X = load('X_6000')
	# save(X, 'X_6000')
	# testX = vectorizeText(data_test['tweet'])

	print 'Vectorized'

	trainData = X[:50000]
	testData = X[50000:]
	trainLabels = dict(s=get('s_raw')[:50000], w=get('w_raw')[:50000], k=get('k_raw')[:50000])
	testLabels = dict(s=get('s_raw')[50000:], w=get('w_raw')[50000:], k=get('k_raw')[50000:])

	trainClasses = genTrainingClasses(trainLabels)

	clf = {}
	predictions = {}

	for i in ('s', 'w', 'k'):
		print 'Starting %s...' % i
		clf[i] = LogisticRegression()
		clf[i].fit(trainData, trainClasses[i])
		predictions[i] = clf[i].predict_proba(testData)

	# writePredictions(predictions)

	predictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)

	labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

	print('S: Train Error', trainError(predictions[:,0:5], labels[:,0:5]))
	print('W: Train Error', trainError(predictions[:,5:10], labels[:,5:10]))
	print('K: Train Error', trainError(predictions[:,10:], labels[:,10:]))
	print('All: Train Error', trainError(predictions, labels))

if __name__ == '__main__':
	main()