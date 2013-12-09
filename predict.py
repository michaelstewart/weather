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
import log
import regressor

MULTIPLIER = 5
# BREAK_POINT = 389730	# Full train
BREAK_POINT = 38973	# 64% Train
# Train
TEST_ERROR = True
TRAIN_ERROR = True
SAVE =  False

# Data
DATA_NAME = 'X_20000'

def main():

	initData()

	# dataList = list(get('tweet')) + list(get('tweet_test'))  # Full Train + Test
	dataList = get('tweet')  # Up to BP Train + From BP Test
	X = vectorizeText(dataList, 100000)
	save(X, 'random100000')
	# X = load('random100000')

	# logPredictions = log.run()
	# regPredictions = regressor.run()

	# save(logPredictions, 'logPredictions')
	# save(regPredictions, 'regPredictions')

	logPredictions = load('logPredictions')
	regPredictions = load('regPredictions')

	actual = np.concatenate((get('s_raw'), get('w_raw'), get('k_raw')), axis=1)

	best = bestAlgo(actual[BREAK_POINT:], logPredictions, regPredictions)
	# Log
	# Data
	trainData = X[BREAK_POINT:] 
	testData = X[:BREAK_POINT]

	print best.shape
	print testData.shape


	clf = trainLogReg(trainData, best, penalty='l2', C=1)

	weights = clf.predict_proba(testData)

	newPredictions = multiplyArr(logPredictions,weights[:,0]) + multiplyArr(regPredictions,weights[:,1])

	combinedPredictions = newPredictions


	if SAVE:
		# Write predictions to file
		writePredictions(combinedPredictions)

	if TEST_ERROR:
		# Calculate Test Error
		testLabels = dict(
			s=get('s_raw')[:BREAK_POINT], 
			w=get('w_raw')[:BREAK_POINT], 
			k=get('k_raw')[:BREAK_POINT]
		)
		labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

		# Calculate the Test Error
		# print('S: Test Error', trainError(predictions['s'], testLabels['s']))
		# print('W: Test Error', trainError(predictions['w'], testLabels['w']))
		# print('K: Test Error', trainError(predictions['k'], testLabels['k']))
		print('All: Test Error', trainError(combinedPredictions, labels))

if __name__ == '__main__':
	main()
