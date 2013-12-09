#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from setup import *
import logging

# logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    filename='logs/svm.log'
# )
# import setupLogging

MULTIPLIER = 5
# BREAK_POINT = 389730	# Full train
# BREAK_POINT = 38973	
BREAK_POINT = 77946

# End
TEST_ERROR = True
TRAIN_ERROR = False
SAVE =  False

# Data
DATA_NAME = 'X_20000'
LOAD = False
SAVE_DATA = False



def main():

	initData()
	disp('Start')

	if LOAD:
		X = load(DATA_NAME)
	else:
		dataList = simpleMultiply(get('tweet'), MULTIPLIER) + simpleMultiply(get('tweet_test'), MULTIPLIER)  # Full Train + Test
		# dataList = simpleMultiply(get('tweet')[:BREAK_POINT], MULTIPLIER) + simpleMultiply(get('tweet')[BREAK_POINT:], MULTIPLIER)  # Up to BP Train + From BP Test
		X = vectorizeText(dataList, 20000)

		if SAVE_DATA:
			save(X, DATA_NAME)

	disp(X.shape)
	disp('Vectorized')

	# Data
	trainData = X[:BREAK_POINT*5]	
	testData = X[BREAK_POINT*5:]

	# trainData = trainData.todense()
	# testData = testData.todense()

	i = 'k'

	classes = multiply(get('%s_raw' % i), MULTIPLIER)
	t0 = time()
	svc = LinearSVC(penalty='l1', dual=False)
	svc.fit(trainData, classes)
	disp('Train time: %d seconds' % (time() - t0))
	save(svc, 'SVC_%s_all' % i, 'pickles/svc/4')
	transArr = svc.transform(X, '3.25*mean')
	save(transArr, 'X_%s_all' % i, 'pickles/svc/4')

	disp(transArr.shape)

	print('End')

if __name__ == '__main__':
	main()

