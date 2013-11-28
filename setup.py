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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from nltk.stem.porter import *


# Train a Model
def trainLogReg(X, y):
    print('_' * 80)
    print("Training: LogisticRegression")
    clf = LogisticRegression(penalty='l1',C=1.25)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    return clf

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

def vectorizeText(data, size):
	# data = cleanData(data)
	# stemmer = PorterStemmer()
	tfidf = TfidfVectorizer(max_features=size, ngram_range = (1,2), strip_accents='unicode', analyzer='word')
	tfidf.fit(data)
	return tfidf.transform(data)

def hashingVector(data, size):
	vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=size)
	return vectorizer.transform(data)

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


def initData():
	global data
	data = {}
	paths = ['data/train.csv', 'data/test.csv']
	data['train'] = p.read_csv(paths[0])
	data['test'] = p.read_csv(paths[1])

	data['tweet'] = np.array(data['train']['tweet'])
	data['tweet_test'] = np.array(data['test']['tweet'])

	data['complete_list'] = np.array(data['train'].ix[:,'s1':'k15'])
	data['s_raw'] = np.array(data['train'].ix[:,'s1':'s5'])
	data['w_raw'] = np.array(data['train'].ix[:,'w1':'w4'])
	data['k_raw'] = np.array(data['train'].ix[:,'k1':'k15'])

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

def get(lookup):
	return data[lookup]

def errorArray(predict, actual):
	return np.array(np.array(predict-actual)**2)

def trainError(predict, actual):
	return np.sqrt(np.sum(errorArray(predict, actual)/ (predict.shape[0]*predict.shape[1])))

def writePredictions(prediction):
	out_rows = np.array(np.hstack([np.matrix(data['test']['id']).T, prediction]))
	col = '%i,' + '%f,'*23 + '%f'
	np.savetxt('output/prediction.csv', out_rows, col, delimiter=',')

def load(filename, folder='pickles'):
	path = folder + '/' + filename + '.p'
	f = open(path, 'rb')
	return pickle.load(f)
	f.close()
	print('Loaded %s' % path)

def save(obj, filename, folder='pickles'):
	f = open(folder + '/' + filename + '.p', 'wb')
	pickle.dump(obj, f)
	f.close()
	print('Saved %s' % filename)
