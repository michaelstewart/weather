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
FOLDER = ''

# Data
DATA_NAME = 'X_20000'

def main():

	initData()

	X = load(DATA_NAME)
	
	print 'Vectorized'

	# Data
	testData = X[BREAK_POINT*MULTIPLIER:]

	print trainData.shape
	print testData.shape

	predictions = {}
	predictionsTrain = {}

	# Sentiment 1-5
	print('Sentiment')
	# save(sentimentClf, 'model_s')
	sentimentClf = load(SMODEL)
	predictions['s'] = sentimentClf.predict_proba(testData)[::5]

	## When 1-4
	print('When')
	whenClf = load('model_w')
	predictions['w'] = whenClf.predict_proba(testData)[::5]

	## Kind 1-15
	print('Kind')

	kindClfs = []
	predictions['k'] = []
	for i in xrange(15):
		kindClfs.append(load('model_k_%d' % i))
		predictions['k'].append(kindClfs[i].predict_proba(testData)[::5,1])

	predictions['k'] = np.array(predictions['k']).T
	combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)


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


if __name__ == '__main__':
	main()
