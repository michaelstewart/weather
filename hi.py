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

def main():

	initData()
	titles = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
	data = get('train');
	for t in titles[9:]:
		x = np.hstack(np.array(data.ix[:,t:t]))

		print '='*80
		print 'For ' + t
		print 'Num of  0: %d' % count((x*5).round(), 0)
		print 'Num of .2: %d' % count((x*5).round(), 1)
		print 'Num of .4: %d' % count((x*5).round(), 2)
		print 'Num of .6: %d' % count((x*5).round(), 3)
		print 'Num of .8: %d' % count((x*5).round(), 4)
		print 'Num of  1: %d' % count((x*5).round(), 5)

		Py.figure()
		# the histogram of the data with histtype='step'
		n, bins, patches = Py.hist(x, 300, range=(0.1,0.9), normed=False, histtype='stepfilled')
		Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)


	Py.show()

def count(arr, num):
	count = 0
	for i in arr:
		count += 1 if i == num else 0
	return count

if __name__ == '__main__':
	main()
