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

MULTIPLIER = 5
# BREAK_POINT = 77946	# Full train
BREAK_POINT = 38973	
TEST_ERROR = False
TRAIN_ERROR = False
SAVE =  False
RUN = True

def lame():
	global trainData, testData

	initData()

	dataList = list(get('tweet')) #+ list(get('tweet_test'))
	X = vectorizeText(dataList, 30000)

	# X = load('Test_X_40000_short')
	# save(X, 'X_k_all', 'pickles/ridge/1')
	
	print 'Vectorized'

	# Data
	trainData = X[:BREAK_POINT]
	testData = X[BREAK_POINT:]

	print trainData.shape
	print testData.shape

	predictions = {}
	predictionsTrain = {}

	# Sentiment 1-5
	print('Sentiment')
	(predictions['s'], predictionsTrain['s']) = learnOverColumns('s', 5)
	predictions['s'] = rowsToOne(predictions['s'])
	predictionsTrain['s'] = rowsToOne(predictionsTrain['s'])

	## When 1-4
	print('When')
	(predictions['w'], predictionsTrain['w']) = learnOverColumns('w', 4)
	predictions['w'] = rowsToOne(predictions['w'])
	predictionsTrain['w'] = rowsToOne(predictionsTrain['w'])
	
	## Kind 1-15
	print('Kind')
	(predictions['k'], predictionsTrain['k']) = learnOverColumns('k', 15)
	predictions['k'] = containMatrix(predictions['k'])
	predictionsTrain['k'] = containMatrix(predictionsTrain['k'])

	combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)

	if RUN == True:
		return combinedPredictions


	if SAVE:
		# Write predictions to file
		writePredictions(combinedPredictions)

	if TEST_ERROR:
		# Calculate Test Error
		testLabels = dict(
			s=get('s_raw')[BREAK_POINT:], 
			w=get('w_raw')[BREAK_POINT:], 
			k=get('k_raw')[BREAK_POINT:]
		)
		labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

		print('S: Test Error', trainError(predictions['s'], testLabels['s']))
		print('W: Test Error', trainError(predictions['w'], testLabels['w']))
		print('K: Test Error', trainError(predictions['k'], testLabels['k']))
		print('All: Test Error', trainError(combinedPredictions, labels))

	if TRAIN_ERROR:
		trainLabels = dict(
			s=get('s_raw')[:BREAK_POINT],
			w=get('w_raw')[:BREAK_POINT],
			k=get('k_raw')[:BREAK_POINT]
		)

		combinedPredictions = np.concatenate((predictionsTrain['s'], predictionsTrain['w'], predictionsTrain['k']), axis=1)

		labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

		# Calculate Train Error
		print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
		print('W: Train Error', trainError(predictionsTrain['w'], trainLabels['w']))
		print('K: Train Error', trainError(predictionsTrain['k'], trainLabels['k']))
		print('All: Train Error', trainError(combinedPredictions, labels))

def learnOverColumns(t, cols):
	# Create class list
	classList = []
	for i in xrange(cols):
		classList.append(get('%s_raw' % t)[:BREAK_POINT,i])

	clfs = []
	predictions = []
	predictionsTrain = []
	for i in xrange(cols):
		print 'Starting %d...' % i
		clfs.append(trainRidgeRegressor(trainData, classList[i]))
		# save(clfs[i], 'all_%s_%d' % (t,i), 'pickles/ridge/1')
		predictions.append(clfs[i].predict(testData))
		predictionsTrain.append(clfs[i].predict(trainData))

	return (np.array(predictions).T, np.array(predictionsTrain).T)

def run():
	RUN = True
	return lame()

if __name__ == '__main__':
	lame()
