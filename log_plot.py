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
import pylab as Py

MULTIPLIER = 5
# BREAK_POINT = 389730	# Full train
BREAK_POINT = 50000	# 64% Train
TEST_ERROR = True
TRAIN_ERROR = False
SAVE =  False

def main():

	initData()

	# dataList = simpleMultiply(get('tweet'), MULTIPLIER) + simpleMultiply(get('tweet_test'), MULTIPLIER)  # Full Train + Test
	# dataList = simpleMultiply(get('tweet')[:BREAK_POINT], MULTIPLIER) + simpleMultiply(get('tweet')[BREAK_POINT:], MULTIPLIER)  # Up to BP Train + From BP Test
	# X_long = vectorizeText(dataList, 40000)

	# X = load('X_10000')
	X_long = load('X_40000_long')
	# save(X_long, 'X_40000_long')
	# testX = vectorizeText(data_test['tweet'])

	
	print 'Vectorized'
	# X_long = np.array(simpleMultiply(X, MULTIPLIER))


	# Data
	trainData = X_long[:BREAK_POINT*5]
	testData = X_long[BREAK_POINT*5:]

	print trainData.shape
	print testData.shape

	predictions = {}

	# Sentiment 1-5
	print('Sentiment')
	# sClasses = multiply(get('s_raw')[:BREAK_POINT], MULTIPLIER)
	# sClasses = weightToClass(get('s_raw')[:BREAK_POINT])
	# sentimentClf = trainLogReg(trainData, sClasses, penalty='l2', C=0.825)
	# save(sentimentClf, 'model_s')
	sentimentClf = load('model_s')

	predictions['s'] = sentimentClf.predict_proba(testData)[::5]

	## When 1-4
	print('When')
	# wClasses = multiply(get('w_raw')[:BREAK_POINT], MULTIPLIER)
	# wClasses = weightToClass(get('w_raw')[:BREAK_POINT])
	# whenClf = trainLogReg(trainData, wClasses, penalty='l2', C=0.7)
	# save(whenClf, 'model_w')
	whenClf = load('model_w')
	predictions['w'] = whenClf.predict_proba(testData)[::5]

	## Kind 1-15
	print('Kind')
	# Create class list
	kClassList = []
	for i in xrange(15):
		kClassList.append(multiplySingle(get('k_raw')[:BREAK_POINT,i], MULTIPLIER))
		# kClassList.append(weightToClassBinary(get('k_raw')[:BREAK_POINT,i]))

	kindClfs = []
	predictions['k'] = []
	for i in xrange(15):
		# kindClfs.append(trainLogReg(trainData, kClassList[i], penalty='l1', C=1))
		# save(kindClfs[i], 'model_k_%d' % i)
		kindClfs.append(load('model_k_%d' % i))
		predictions['k'].append(kindClfs[i].predict_proba(testData)[::5,1])


	predictions['k'] = np.array(predictions['k']).T
	combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)


	if SAVE:
		# Write predictions to file
		writePredictions(combinedPredictions)
	else:
		if TEST_ERROR:
			# Calculate Test Error
			testLabels = dict(
				s=get('s_raw')[BREAK_POINT:], 
				w=get('w_raw')[BREAK_POINT:], 
				k=get('k_raw')[BREAK_POINT:]
			)
			labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

			i = 0

			Py.figure()
			# the histogram of the data
			n, bins, patches = Py.hist(abs(roundToNearest(combinedPredictions,5)-labels), 300, normed=False, histtype='stepfilled')
			Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)

			# Py.figure()
			# # the histogram of the data
			# n, bins, patches = Py.hist(testLabels['s'][:,i], 300, range=(0,1), normed=False, histtype='stepfilled')
			# Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)

			# Py.figure()
			# # the histogram of the data
			# n, bins, patches = Py.hist(roundToNearest(predictions['s'][:,0], 5), 300, range=(0,1), normed=False, histtype='stepfilled')
			# Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)

			# Py.figure()
			# # the histogram of the data
			# n, bins, patches = Py.hist(roundToNearest(testLabels['s'][:,0], 5), 300, range=(0,1), normed=False, histtype='stepfilled')
			# Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)

			

			# print('S%d: Test Error' % i, trainError(predictions['s'][:,i:i+1], testLabels['s'][:,i:i+1]))

			print('S: Test Error', trainError(predictions['s'], testLabels['s']))
			print('W: Test Error', trainError(predictions['w'], testLabels['w']))
			print('K: Test Error', trainError(predictions['k'], testLabels['k']))
			print('All: Test Error', trainError(combinedPredictions, labels))

			Py.show()

		if TRAIN_ERROR:
			trainLabels = dict(
				s=get('s_raw')[:BREAK_POINT], 
				w=get('w_raw')[:BREAK_POINT], 
				k=get('k_raw')[:BREAK_POINT:]
			)
			labels = np.concatenate((trainLabels['s'], trainLabels['w'], trainLabels['k']), axis=1)

			# Calculate Train Error
			print('S: Train Error', trainError(predictions['s'], testLabels['s']))
			print('W: Train Error', trainError(predictions['w'], testLabels['w']))
			print('K: Train Error', trainError(predictions['k'], testLabels['k']))
			print('All: Train Error', trainError(combinedPredictions, labels))

	

if __name__ == '__main__':
	main()
