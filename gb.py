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
import argparse
import logging

args = parseArguments()

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
   filename='logs/gb2-%s%d.log' % (args.type[0], args.index[0])
)
import setupLogging

MULTIPLIER = 5
# BREAK_POINT = 389730	# Full train
BREAK_POINT = 38973	
TEST_ERROR = False
TRAIN_ERROR = False
SAVE =  False

# Data
DATA_NAME = 'X_20000'
LOAD = True
SAVE_DATA = False

def main():

	
	initData()
	print('Start')

	X = load(args.loadFile[0], args.loadFolder[0])

	print(X.shape)
	print('Vectorized')

	# Data
	trainData = X[:BREAK_POINT*5]
	testData = X[BREAK_POINT*5:]

	trainData = trainData.todense()
	testData = testData.todense()

	print('Densified')

	if args.type[0] == 's':

		# Sentiment 1-5
		print('Sentiment')
		classes = multiply(get('s_raw')[:BREAK_POINT], MULTIPLIER)
		# sClasses = weightToClass(get('s_raw')[:BREAK_POINT])
		clf = trainGB(trainData, classes, n_estimators=200)
		save(clf, 's', args.saveFolder[0])

	if args.type[0] == 'w':

		# When 1-4
		print('When')
		classes = multiply(get('w_raw')[:BREAK_POINT], MULTIPLIER)
		clf = trainGB(trainData, classes, n_estimators=200)
		save(clf, 'w', args.saveFolder[0])

	if args.type[0] == 'k':
		## Kind 1 - 15
		i = args.index[0]
		classes = multiplySingle(get('k_raw')[:BREAK_POINT,i], MULTIPLIER)
		clf = trainGB(trainData, classes, n_estimators=400)
		save(clf, 'k%d' % i, args.saveFolder[0])


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
				# k=get('k_raw')[:BREAK_POINT]
			)
			# labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

			# Calculate Train Error
			print('S: Train Error', trainError(predictionsTrain['s'], trainLabels['s']))
			plogging.info('S: Train Error' + str(trainError(predictionsTrain['s'], trainLabels['s'])))

			# print('W: Train Error', trainError(predictions['w'], testLabels['w']))
			# print('K: Train Error', trainError(predictions['k'], testLabels['k']))
			# print('All: Train Error', trainError(combinedPredictions, labels))


if __name__ == '__main__':
	main()
