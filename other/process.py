#!/usr/bin/python

import csv

filename = 'prediction.csv'

f = open(filename)
o = open('processed-' + filename, 'w')
# f.readline() # skip firstline

o.write('id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15\n')

over = 0
under = 0
for row in csv.reader(f, delimiter=',', skipinitialspace=True):
	nrow = [row[0]]
	for i in row[1:]:
		if float(i) < 0: 
			nrow.append('0')
			under += 1
		if float(i) > 1:
			nrow.append('1')
			over += 1
		else: nrow.append(i)
	o.write(','.join(nrow) + '\n')

print over
print under

f.close()
o.close()

