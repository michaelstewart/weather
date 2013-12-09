#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from setup import *
import logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
   filename='logs/rf.log'
)
import setupLogging

MULTIPLIER = 5
# BREAK_POINT = 389730	# Full train
BREAK_POINT = 50000	
TEST_ERROR = True
TRAIN_ERROR = True
SAVE =  False

# Data
DATA_NAME = 'X_20000'
LOAD = True
SAVE_DATA = False

def main():

	initData()
	disp('Start')

	if LOAD:
		X = load(DATA_NAME)
	else:
		# dataList = simpleMultiply(get('tweet'), MULTIPLIER) + list(get('tweet_test'))  # Full Train + Test
		dataList = simpleMultiply(get('tweet')[:BREAK_POINT], MULTIPLIER) + simpleMultiply(get('tweet')[BREAK_POINT:], MULTIPLIER)  # Up to BP Train + From BP Test
		X = vectorizeText(dataList, 20000)

		if SAVE_DATA:
			save(X, DATA_NAME)

	disp(X.shape)
	disp('Vectorized')

	# Data
	trainData = X_long[:BREAK_POINT*5]
	testData = X_long[BREAK_POINT*5:]

	trainData = trainData.todense()
	testData = testData.todense()

	predictions = {}

	# Sentiment 1-5
	print('Sentiment')
	sClasses = multiply(get('s_raw')[:BREAK_POINT], MULTIPLIER)
	# sClasses = weightToClass(get('s_raw')[:BREAK_POINT])
	sentimentClf = trainRF(trainData, sClasses)
	# save(sentimentClf, 'model_s')
	# sentimentClf = load('model_s')

	predictions['s'] = sentimentClf.predict_proba(testData)[::5]
	predictionsTrain['s'] = sentimentClf.predict_proba(testData)[::5]

	## When 1-4
	print('When')
	# wClasses = multiply(get('w_raw')[:BREAK_POINT], MULTIPLIER)
	# # wClasses = weightToClass(get('w_raw')[:BREAK_POINT])
	# whenClf = trainRF(trainData, wClasses)
	# # save(whenClf, 'model_w_RF_2')
	# # whenClf = load('model_w')
	# predictions['w'] = whenClf.predict_proba(testData)

	## Kind 1-15
	# print('Kind')
	# # Create class list
	# kClassList = []
	# for i in xrange(15):
	# 	kClassList.append(multiplySingle(get('k_raw')[:BREAK_POINT,i], MULTIPLIER))
	# 	# kClassList.append(weightToClassBinary(get('k_raw')[:BREAK_POINT,i]))

	# kindClfs = []
	# predictions['k'] = []
	# for i in xrange(15):
	# 	kindClfs.append(trainLogReg(trainData, kClassList[i]))
	# 	save(kindClfs[i], 'model_k_%d' % i)
	# 	predictions['k'].append(kindClfs[i].predict_proba(testData)[:,1])


	# predictions['k'] = np.array(predictions['k']).T
	# combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)


	if SAVE:
		# Write predictions to file
		writePredictions(combinedPredictions)
	else:
		if TEST_ERROR:
			# Calculate Test Error
			testLabels = dict(
				s=get('s_raw')[BREAK_POINT:], 
				# w=get('w_raw')[BREAK_POINT:], 
				# k=get('k_raw')[BREAK_POINT:]
			)
			# labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

			print('S: Test Error', trainError(predictions['s'], testLabels['s']))
			logging.info('S: Test Error ' + str(trainError(predictions['s'], testLabels['s'])))

			# print('W: Test Error', trainError(predictions['w'], testLabels['w']))
			# logging.info('W: Test Error' + str(trainError(predictions['w'], testLabels['w'])))
			# print('K: Test Error', trainError(predictions['k'], testLabels['k']))
			# print('All: Test Error', trainError(combinedPredictions, labels))

		if TRAIN_ERROR:
			trainLabels = dict(
				s=get('s_raw')[:BREAK_POINT], 
				# w=get('w_raw')[:BREAK_POINT], 
				# k=get('k_raw')[:BREAK_POINT:]
			)
			# labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

			# Calculate Train Error
			print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
			plogging.info('S: Train Error' + str(trainError(predictionsTrain['s'], trainLabels['s'])))

			# print('W: Train Error', trainError(predictions['w'], testLabels['w']))
			# print('K: Train Error', trainError(predictions['k'], testLabels['k']))
			# print('All: Train Error', trainError(combinedPredictions, labels))


	save(sentimentClf, 'model_s')

if __name__ == '__main__':
	main()
