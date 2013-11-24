#!/usr/bin/python

import csv

filename = 'prediction.csv'

f = open(filename)
o = open('processed-' + filename, 'w')
# f.readline() # skip firstline

for row in csv.reader(f, delimiter=',', skipinitialspace=True):
	nrow = []
	for i in row:
		if float(i) < 0: nrow.append('0')			
		else: nrow.append(i)
	o.write(','.join(nrow) + '\n')

f.close()
o.close()

