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
import logging

# logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    filename='logs/rfregressor.log'
# )
# import setupLogging

MULTIPLIER = 5
# BREAK_POINT = 77946	# Full train
BREAK_POINT = 50000	# 64% Train
TEST_ERROR = True
TRAIN_ERROR = True
SAVE =  False

def main():
	global trainData, testData

	initData()

	# dataList = simpleMultiply(get('tweet'), MULTIPLIER) + simpleMultiply(get('tweet_test'), MULTIPLIER)  # Full Train + Test
	# dataList = list(get('tweet'))
	# X = vectorizeText(dataList, 20000)

	X = load('X_w', 'pickles/svc/2')
	# save(X, 'Test_X_40000_short')
	
	print 'Vectorized'

	# Data
	trainData = X[:BREAK_POINT].todense()
	testData = X[BREAK_POINT:].todense()

	print trainData.shape
	print testData.shape

	predictions = {}
	predictionsTrain = {}

	# Sentiment 1-5
	# print('Sentiment')
	# (predictions['s'], predictionsTrain['s']) = learnOverColumns('s', 5)
	# predictions['s'] = rowsToOne(predictions['s'])
	# predictionsTrain['s'] = rowsToOne(predictionsTrain['s'])

	## When 1-4
	print('When')
	(predictions['w'], predictionsTrain['w']) = learnOverColumns('w', 4)
	predictions['w'] = rowsToOne(predictions['w'])
	predictionsTrain['w'] = rowsToOne(predictionsTrain['w'])

	print predictions['w']
	
	## Kind 1-15
	# print('Kind')
	# (predictions['k'], predictionsTrain['k']) = learnOverColumns('k', 15)
	# predictions['k'] = containMatrix(predictions['k'])
	# predictionsTrain['k'] = containMatrix(predictionsTrain['k'])

	# combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)


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
		# labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

		print predictions['w'].shape
		print testLabels['w'].shape
		# print('S: Test Error', trainError(predictions['s'], testLabels['s']))
		print('W: Test Error', trainError(predictions['w'], testLabels['w']))
		# print('K: Test Error', trainError(predictions['k'], testLabels['k']))
		# print('All: Test Error', trainError(combinedPredictions, labels))

	if TRAIN_ERROR:
		trainLabels = dict(
			s=get('s_raw')[:BREAK_POINT],
			w=get('w_raw')[:BREAK_POINT],
			k=get('k_raw')[:BREAK_POINT]
		)

		# combinedPredictions = np.concatenate((predictionsTrain['s'], predictionsTrain['w'], predictionsTrain['k']), axis=1)

		# labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

		# Calculate Train Error
		# print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
		print('W: Train Error', trainError(predictionsTrain['w'], trainLabels['w']))
		# print('K: Train Error', trainError(predictionsTrain['k'], trainLabels['k']))
		# print('All: Train Error', trainError(combinedPredictions, labels))

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
		# clfs.append(trainRFRegressor(trainData, classList[i]))
		# save(clfs[i], '%s%d' % (t, i), 'pickles/rf/')
		clfs.append(load('%s%d' % (t, i), 'pickles/rf/1'))
		# kindClfs.append(load('model_k_%d' % i))
		predictions.append(clfs[i].predict(testData))
		predictionsTrain.append(clfs[i].predict(trainData))

	return (np.array(predictions).T, np.array(predictionsTrain).T)

if __name__ == '__main__':
	main()
