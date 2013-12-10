#!/usr/bin/python
import pickle
# import pandas as p
import numpy as np
# import nltk
# from setup import *
import pylab as Py


def load(filename, folder='pickles'):
	path = folder + '/' + filename + '.p'
	f = open(path, 'rb')
	return pickle.load(f)
	f.close()
	disp('Loaded %s' % path)


def main():

	if False:
		initData()
		titles = ['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']
		data = get('train');

		x = np.hstack(np.array(data.ix[:,'s2':'s2']))
		save(x, 'hist_1')
		exit()

	else:
		x = load('hist_1')	
		Py.figure()
		# the histogram of the data with histtype='step'
		n, bins, patches = Py.hist(x, 300, range=(0.1,0.9), normed=True, histtype='stepfilled')
		Py.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
		Py.title('Negative Sentiment Label Frequency')
		Py.xlabel('Negative Sentiment Label') 
		Py.ylabel('Frequency')
		Py.show()

if __name__ == '__main__':
	main()

