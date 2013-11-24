#!/usr/bin/python
import pickle
import pandas as p
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor

def main():

	initData()

	X = vectorizeText(data_train['tweet'])
	# X = load('X_100')
	# save(X, 'X_1000')
	testX = vectorizeText(data_test['tweet'])

	print 'Vectorized'

	trainData = X
	testData = testX
	trainLabels = dict(s=s_list, w=w_list, k=k_list)
	testLabels = dict(s=s_list_test, w=w_list_test, k=k_list_test)

	trainClasses = genTrainingClasses(trainLabels)

	# Sentiment
	models = {}
	predictions = {}

	for i in ('s', 'w'):
		print 'Training on: ' + i
		models[i] = linear_model.LogisticRegression()
		models[i].fit(trainData, trainClasses[i])
		predictions[i] = models[i].predict_proba(testData)

	print 'Training on: k'
	models['k'] = []
	predictions['k'] = []
	for i in xrange(15):
		print i+1
		temp = load('run_5_model_k_' + str(i), 'downloadedpickles')
		models['k'].append(temp)
		predictions['k'].append(temp.predict(testData))

	predictions['k'] = np.column_stack(predictions['k'])

	joined_predictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)

	print joined_predictions

	writePredictions(joined_predictions)

	# labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

	# print('S: Train Error', trainError(predictions['s'], testLabels['s']))
	# print('W: Train Error', trainError(predictions['w'], testLabels['w']))
	# print('K: Train Error', trainError(predictions['k'], testLabels['k']))
	# print('All: Train Error', trainError(joined_predictions, labels))

def weightToClass(features):
	l = []
	for feature in features:
		l.append(feature.argmax())
	return l

def genTrainingClasses(labels):
	resultDict = {}
	for (k, fList) in labels.iteritems():
		resultDict[k] = weightToClass(fList)
	return resultDict

def vectorizeText(data):
	# data = cleanData(data)
	tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
	tfidf.fit(data)
	return tfidf.transform(data)

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

def initData():
	global data_train, data_test, s_list, w_list, k_list, s_list_test, w_list_test, k_list_test
	paths = ['data/train.csv', 'data/test.csv']
	data_train = p.read_csv(paths[0])
	data_test = p.read_csv(paths[1])

	s_list = np.array(data_train.ix[:,'s1':'s5'])
	w_list = np.array(data_train.ix[:,'w1':'w4'])
	k_list = np.array(data_train.ix[:,'k1':'k15'])

	s_list_test = np.array(data_train.ix[:,'s1':'s5'])
	w_list_test = np.array(data_train.ix[:,'w1':'w4'])
	k_list_test = np.array(data_train.ix[:,'k1':'k15'])

def trainError(predict, actual):
	return np.sqrt(np.sum(np.array(np.array(predict-actual)**2)/ (predict.shape[0]*24.0)))

def writePredictions(prediction):
	out_rows = np.array(np.hstack([np.matrix(data_test['id']).T, prediction])) 
	col = '%i,' + '%f,'*23 + '%f'
	np.savetxt('output/prediction.csv', out_rows, col, delimiter=',')

def load(filename, folder='pickles'):
	path = folder + '/' + filename + '.p'
	print path
	f = open(path, 'rb')
	return pickle.load(f)
	f.close()
	print('Loaded %s' % filename)

def save(obj, filename, folder='pickles'):
	f = open(folder + '/' + filename + '.p', 'wb')
	pickle.dump(obj, f)
	f.close()
	print('Saved %s' % filename)

if __name__ == '__main__':
	main()