#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import logging
import argparse



####### Training Models #######

def trainLogReg(X, y, penalty='l2', C=1):
    disp('=' * 40)
    disp("Training: LogisticRegression")
    clf = LogisticRegression(penalty=penalty,C=C)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    disp("train time: %0.3fs" % train_time)
    return clf

def trainNB(X, y):
    disp('=' * 40)
    disp("Training: MultinomialNB")
    clf = MultinomialNB()
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    disp("train time: %0.3fs" % train_time)
    return clf

def trainGB(X, y, n_estimators=100, learning_rate=0.1):
    disp('=' * 40)
    disp("Training: GradientBoostingClassifier")
    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, subsample=0.5, max_depth=3, verbose=1)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    disp("train time: %0.3fs" % train_time)
    return clf

def trainRF(X, y):
    disp('=' * 40)
    disp("Training: RandomForestClassifier")
    clf = RandomForestClassifier(n_estimators=1, verbose=1, n_jobs=-1)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    disp("train time: %0.3fs" % train_time)
    return clf

def trainRFRegressor(X, y):
    disp('=' * 40)
    disp("Training: RandomForestRegressor")
    clf = RandomForestRegressor(n_estimators=1, verbose=1, n_jobs=12)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    disp("train time: %0.3fs" % train_time)
    return clf

def trainSGD(X, y, penalty='l2', C=1, n_iter=50):
	print('=' * 40)
	print("Training: SGDClassifier")
	clf = SGDClassifier(loss='log', penalty='l2', n_iter=n_iter, shuffle=True, n_jobs=-1)
	t0 = time()
	clf.fit(X, y)
	train_time = time() - t0
	disp("train time: %0.3fs" % train_time)
	return clf

def trainSGDRegressor(X, y, penalty='l2', C=1, n_iter=600):
	print('=' * 40)
	print("Training: SGDRegressor")
	clf = SGDRegressor(n_iter=n_iter)
	t0 = time()
	clf.fit(X, y)
	train_time = time() - t0
	disp("train time: %0.3fs" % train_time)
	return clf

def trainRidgeRegressor(X, y, penalty='l2', C=1, n_iter=300, ):
	print('=' * 40)
	print("Training: Ridge")
	clf = Ridge(alpha=1.75)
	t0 = time()
	clf.fit(X, y)
	train_time = time() - t0
	disp("train time: %0.3fs" % train_time)
	return clf

###### TFIDF ######

def vectorizeText(data, size):
	tfidf = TfidfVectorizer(max_features=size, ngram_range = (1,2), min_df=3,
		strip_accents='unicode', analyzer='word')
	tfidf.fit(data)
	return tfidf.transform(data)

def vectorizeTextC(data, size):
	cv = CountVectorizer(max_features=size, min_df=3, ngram_range = (1,2),
		strip_accents='unicode', analyzer='word', binary=True)
	cv.fit(data)
	return cv.transform(data)

def hashingVector(data, size):
	vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=size)
	return vectorizer.transform(data)


###### Data Manipulation ######

def initData():
	global data
	data = {}
	paths = ['data/train.csv', 'data/test.csv']
	data['train'] = p.read_csv(paths[0])
	data['test'] = p.read_csv(paths[1])

	data['tweet'] = np.array(data['train']['tweet'])
	data['tweet'] = appendLocation(data['tweet'], np.array(data['train'].ix[:,'state']))
	data['tweet'] = appendLocation(data['tweet'], np.array(data['train'].ix[:,'location']))
	data['tweet_test'] = np.array(data['test']['tweet'])
	data['tweet_test'] = appendLocation(data['tweet_test'], np.array(data['test'].ix[:,'state']))
	data['tweet_test'] = appendLocation(data['tweet_test'], np.array(data['test'].ix[:,'location']))

	data['train_sentF'] = p.read_csv('data/trainSent.csv')
	data['train_sent'] = np.array(data['train_sentF'].ix[:,'sent'])

	data['complete_list'] = np.array(data['train'].ix[:,'s1':'k15'])
	data['s_raw'] = np.array(data['train'].ix[:,'s1':'s5'])
	data['w_raw'] = np.array(data['train'].ix[:,'w1':'w4'])
	data['k_raw'] = np.array(data['train'].ix[:,'k1':'k15'])

def appendLocation(arr1, arr2):
	for i in xrange(arr1.shape[0]):
		arr1[i] += ' ' + str(arr2[i])
	return arr1

def multiply(arr, multiplier):
	l = []
	for i in arr:
		# loop over each instance
		adds = 0
		for (j_index, j) in enumerate(map(int,(i*multiplier).round())):
			#loop over the frequencies of each feature
			for k in xrange(j):
				if adds < multiplier:
					l.append(j_index)
				adds += 1
		while adds < multiplier:
			l.append(i.argmax())
			adds += 1
	return np.array(l)

def multiplySingle(arr, multiplier):
	l = []
	for i in arr:
		# loop over each instance
		adds = 0
		for j in xrange(int(round(i*multiplier))):
			l.append(1)
			adds += 1
		while adds < multiplier:
			l.append(0)
			adds += 1
	return np.array(l)

def simpleMultiply(arr, multiplier):
	l = []
	for i in arr:
		for j in xrange(multiplier):
			l.append(i)
	return l

def containList(l):
	for i in xrange(len(l)):
		if l[i] > 1:
			l[i] = 1
		elif l[i] < 0:
			l[i] = 0
	return l

def containMatrix(m):
	for i in xrange(m.shape[0]):
		m[i,:] = containList(m[i,:])
	return m

def rowsToOne(arr):
	l = []
	for i in arr:
		i = containList(i)
		i = i/i.sum()
		l.append(np.array(i))
	return np.array(l)


####### Helpers #######
def get(lookup):
	return data[lookup]

def errorArray(predict, actual):
	return np.array(np.array(predict-actual)**2)

def trainError(predict, actual):
	return np.sqrt(np.sum(errorArray(predict, actual))/ (predict.shape[0]*predict.shape[1]))

def bestAlgo(actual, predict1, predict2):
	best = []
	for i in xrange(actual.shape[0]):
		errors = []
		errors.append(np.sqrt(np.sum(errorArray(predict1[i], actual[i]))/ (predict1.shape[0])))
		errors.append(np.sqrt(np.sum(errorArray(predict2[i], actual[i]))/ (predict2.shape[0])))
		best.append(np.array(errors).argmin())
	return np.array(best)

def multiplyArr(arr, mult):
	for i in xrange(arr.shape[0]):
		arr[i] = arr[i]*mult[i]
	return arr

def writePredictions(prediction):
	out_rows = np.array(np.hstack([np.matrix(data['test']['id']).T, prediction]))
	col = '%i,' + '%f,'*23 + '%f'
	np.savetxt('output/prediction.csv', out_rows, col, delimiter=',')

def load(filename, folder='pickles'):
	path = folder + '/' + filename + '.p'
	f = open(path, 'rb')
	return pickle.load(f)
	f.close()
	disp('Loaded %s' % path)

def save(obj, filename, folder='pickles'):
	f = open(folder + '/' + filename + '.p', 'wb')
	pickle.dump(obj, f)
	f.close()
	disp('Saved %s' % filename)

def disp(string):
	print(string)
	# logging.info(string)

def parseArguments(used='gb'):
	parser = argparse.ArgumentParser()
	if used == 'gb':
		parser.add_argument('type', type=str, nargs=1)
		parser.add_argument('index', type=int, nargs=1)
		parser.add_argument('loadFile', type=str, nargs=1)
		parser.add_argument('loadFolder', type=str, nargs=1)
		parser.add_argument('saveFile', type=str, nargs=1)
		parser.add_argument('saveFolder', type=str, nargs=1)
	elif used == 'predict':
		parser.add_argument('data', type=str, nargs=1)
		parser.add_argument('dataFolder', type=int, nargs=1)
		parser.add_argument('loadFile', type=str, nargs=1)
		parser.add_argument('loadFolder', type=str, nargs=1)
		parser.add_argument('saveFile', type=str, nargs=1)
		parser.add_argument('saveFolder', type=str, nargs=1)

	return parser.parse_args()


###### OLD ######

def cleanTweet(text):
    l = []
    data = nltk.word_tokenize(text)
    for i in data:
          l.append(i.rstrip('\n'))
    return " ".join(l)

def cleanData(rows):
	for i in xrange(len(rows)):
		rows[i] = cleanTweet(rows[i])
	return rows

def removeOutOfBounds(arr):
	for x in np.nditer(arr, op_flags=['readwrite']):
		if x < 0:
			x = 0
		elif x > 1:
			x = 1
	return arr

def weightToClass(features):
	l = []
	for feature in features:
		l.append(feature.argmax())
	return l

def weightToClassBinary(features):
	l = []
	for feature in features:
		boundary = 0.5
		if feature > boundary:
			l.append(1)
		else:
			l.append(0)
	return l

def genTrainingClasses(labels):
	resultDict = {}
	for (k, fList) in labels.iteritems():
		resultDict[k] = weightToClass(fList)
	return resultDict

def newWeightToClass(features):
	l = []
	for feature in features:
		l.append(feature.argmax())
	return l

def newTrainingClasses(labels):
	resultDict = {}
	for (k, fList) in labels.iteritems():
		resultDict[k] = weightToClass(fList)
	return resultDict

def roundToNearest(arr, multiplier):
	l = []
	for i in arr:
		l.append((i*multiplier).round()/5)
	return np.array(l)




