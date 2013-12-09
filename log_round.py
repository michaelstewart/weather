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
TEST_ERROR = True
TRAIN_ERROR = True
SAVE =  False

def main():

	initData()

	# dataList = simpleMultiply(get('tweet'), MULTIPLIER) + simpleMultiply(get('tweet_test'), MULTIPLIER)  # Full Train + Test
	dataList = simpleMultiply(get('tweet')[:BREAK_POINT], MULTIPLIER) + simpleMultiply(get('tweet')[BREAK_POINT:], MULTIPLIER)  # Up to BP Train + From BP Test
	X = vectorizeText(dataList, 40000)

	# X = load('X_100000')
	save(X, 'X_100000')
	
	print 'Vectorized'

	# Data
	trainData = X[:BREAK_POINT*MULTIPLIER]
	testData = X[BREAK_POINT*MULTIPLIER:]

	print trainData.shape
	print testData.shape

	predictions = {}
	predictionsTrain = {}

	# Sentiment 1-5
	print('Sentiment')
	sClasses = multiply(get('s_raw')[:BREAK_POINT], MULTIPLIER)
	sentimentClf = trainLogReg(trainData, sClasses, penalty='l2', C=0.825)
	# save(sentimentClf, 'model_s')
	# sentimentClf = load('model_s')
	predictions['s'] = sentimentClf.predict_proba(testData)[::5]
	predictionsTrain['s'] = sentimentClf.predict_proba(trainData)[::5]

	## When 1-4
	print('When')
	wClasses = multiply(get('w_raw')[:BREAK_POINT], MULTIPLIER)
	whenClf = trainLogReg(trainData, wClasses, penalty='l2', C=0.7)
	# save(whenClf, 'model_w')
	# whenClf = load('model_w')
	predictions['w'] = whenClf.predict_proba(testData)[::5]
	predictionsTrain['w'] = whenClf.predict_proba(trainData)[::5]

	## Kind 1-15
	print('Kind')
	# Create class list
	kClassList = []
	for i in xrange(15):
		kClassList.append(multiplySingle(get('k_raw')[:BREAK_POINT,i], MULTIPLIER))

	kindClfs = []
	predictions['k'] = []
	predictionsTrain['k'] = []
	for i in xrange(15):
		kindClfs.append(trainLogReg(trainData, kClassList[i], penalty='l1', C=1))
		# save(kindClfs[i], 'model_k_%d' % i)
		# kindClfs.append(load('model_k_%d' % i))
		predictions['k'].append(kindClfs[i].predict_proba(testData)[::5,1])
		predictionsTrain['k'].append(kindClfs[i].predict_proba(trainData)[::5,1])


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

	if TRAIN_ERROR:
		trainLabels = dict(
			s=get('s_raw')[:BREAK_POINT],
			w=get('w_raw')[:BREAK_POINT],
			k=get('k_raw')[:BREAK_POINT]
		)

		predictionsTrain['k'] = np.array(predictionsTrain['k']).T			
		combinedPredictions = np.concatenate((predictionsTrain['s'], predictionsTrain['w'], predictionsTrain['k']), axis=1)

		labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

		# Calculate Train Error
		print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
		print('W: Train Error', trainError(predictionsTrain['w'], trainLabels['w']))
		print('K: Train Error', trainError(predictionsTrain['k'], trainLabels['k']))
		print('All: Train Error', trainError(combinedPredictions, labels))

	

if __name__ == '__main__':
	main()
