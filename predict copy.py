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
# BREAK_POINT = 389730	# Full train
BREAK_POINT = 50000	# 64% Train
# Train
TEST_ERROR = True
TRAIN_ERROR = True
SAVE =  False

#Models
SMODEL = ''
WMODEL = ''
KBASEMODEL = ''
FOLDER = 'pickles/gb/3'
FOLDER1 = 'pickles/gb/4'

# Data
DATA_NAME = 'X_20000'

def main():

	initData()

	Xs = load('X_s', 'pickles/svc/1')
	Xk = load('X_k', 'pickles/svc/1')

	print 'Vectorized'

	# Data
	testData = {}
	trainData = {}
	testData['s'] = Xs[BREAK_POINT*MULTIPLIER:].todense()
	trainData['s'] = Xs[:BREAK_POINT*MULTIPLIER].todense()
	testData['k'] = Xk[BREAK_POINT*MULTIPLIER:].todense()
	trainData['k'] = Xk[:BREAK_POINT*MULTIPLIER].todense()


	predictions = {}
	predictionsTrain = {}

	# Sentiment 1-5
	print('Sentiment')
	# save(sentimentClf, 'model_s')
	sentimentClf = load('s', FOLDER)
	predictions['s'] = sentimentClf.predict_proba(testData['s'])[::5]
	predictionsTrain['s'] = sentimentClf.predict_proba(trainData['s'])[::5]



	# ## When 1-4
	# print('When')
	# whenClf = load('model_w')
	# predictions['w'] = whenClf.predict_proba(testData)[::5]

	## Kind 1-15
	print('Kind')

	kClfs = []
	predictions['k'] = []
	predictionsTrain['k'] = []
	for i in xrange(15):
		kClfs.append(load('k%d' % i, FOLDER1))
		predictions['k'].append(kClfs[i].predict_proba(testData['k'])[::5,1])
		predictionsTrain['k'].append(kClfs[i].predict_proba(trainData['k'])[::5,1])

	predictions['k'] = np.array(predictions['k']).T
	predictionsTrain['k'] = np.array(predictionsTrain['k']).T
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

		# Calculate the Test Error
		print('S: Test Error', trainError(predictions['s'], testLabels['s']))
		# print('W: Test Error', trainError(predictions['w'], testLabels['w']))
		print('K: Test Error', trainError(predictions['k'], testLabels['k']))
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
		print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
		# print('W: Train Error', trainError(predictionsTrain['w'], trainLabels['w']))
		print('K: Train Error', trainError(predictionsTrain['k'], trainLabels['k']))
		# print('All: Train Error', trainError(combinedPredictions, labels))


if __name__ == '__main__':
	main()
