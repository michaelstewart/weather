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
BREAK_POINT = 389730

def main():

	initData()

	# X = vectorizeText(get('tweet')[BREAK_POINT:], 10000)
	dataList = simpleMultiply(get('tweet'), MULTIPLIER) + list(get('tweet_test'))
	X_long = vectorizeText(dataList, 10000)

	# X = load('X_10000')
	# X_long = load('X_10000_long_t1')
	# save(X, 'X_10000_t1')
	save(X_long, 'X_10000_long_t2')
	# testX = vectorizeText(data_test['tweet'])

	
	print 'Vectorized'
	# X_long = np.array(simpleMultiply(X, MULTIPLIER))
	print X_long.shape

	# Data
	trainData = X_long[:BREAK_POINT]
	testData = X_long[BREAK_POINT:]


	# Labels
	# trainLabels = dict(
	# 	s=get('s_list')[:BREAK_POINT], 
	# 	w=get('w_list')[:BREAK_POINT], 
	# 	k=get('k_list')[:BREAK_POINT]
	# 	)
	# testLabels = dict(
	# 	s=get('s_raw')[BREAK_POINT:], 
	# 	w=get('w_raw')[BREAK_POINT:], 
	# 	k=get('k_raw')[BREAK_POINT:]
	# 	)

	predictions = {}

	# Sentiment 1-5
	print('Sentiment')
	sClasses = multiply(get('s_raw'), MULTIPLIER)
	# sClasses = weightToClass(get('s_raw')[:BREAK_POINT])
	sentimentClf = trainLogReg(trainData, sClasses)
	save(sentimentClf, 'model_s')
	# sentimentClf = load('model_s')

	predictions['s'] = sentimentClf.predict_proba(testData)

	## When 1-4
	print('When')
	wClasses = multiply(get('w_raw'), MULTIPLIER)
	# wClasses = weightToClass(get('w_raw')[:BREAK_POINT])
	whenClf = trainLogReg(trainData, wClasses)
	save(whenClf, 'model_w')
	# whenClf = load('model_w')
	predictions['w'] = whenClf.predict_proba(testData)

	## Kind 1-15
	print('Kind')
	# Create class list
	kClassList = []
	for i in xrange(15):
		kClassList.append(multiplySingle(get('k_raw')[:,i], MULTIPLIER))
		# kClassList.append(weightToClassBinary(get('k_raw')[:BREAK_POINT,i]))

	kindClfs = []
	predictions['k'] = []
	for i in xrange(15):
		kindClfs.append(trainLogReg(trainData, kClassList[i]))
		save(kindClfs[i], 'model_k_%d' % i)
		predictions['k'].append(kindClfs[i].predict_proba(testData)[:,1])


	predictions['k'] = np.transpose(np.array(predictions['k']))
	combinedPredictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)
	writePredictions(combinedPredictions)

	# labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

	# print('S: Train Error', trainError(predictions['s'], testLabels['s']))
	# print('W: Train Error', trainError(predictions['w'], testLabels['w']))
	# print('K: Train Error', trainError(predictions['k'], testLabels['k']))
	# print('All: Train Error', trainError(combinedPredictions, labels))

if __name__ == '__main__':
	main()
