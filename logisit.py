
def main_log():

	initData()

	# X = vectorizeText(data_train['tweet'])
	X = load('X')
	# save(X, 'tok_s')
	# testX = vectorizeText(data_test['tweet'])

	print 'Vectorized'

	trainData = X[:50000]
	testData = X[50000:]
	trainLabels = dict(s=s_list[:50000], w=w_list[:50000], k=k_list[:50000])
	testLabels = dict(s=s_list[50000:], w=w_list[50000:], k=k_list[50000:])

	trainClasses = genTrainingClasses(trainLabels)

	# Sentiment
	clf = {}
	predictions = {}

	for i in ('s', 'w', 'k'):
		print 'Starting ' + i + ' ...',
		clf[i] = linear_model.LogisticRegression()
		clf[i].fit(trainData, trainClasses[i])
		predictions[i] = clf[i].predict_proba(testData)
		print ' Done'

	joined_predictions = np.concatenate((predictions['s'], predictions['w'], predictions['k']), axis=1)

	# writePredictions(predictions)



	labels = np.concatenate((testLabels['s'], testLabels['w'], testLabels['k']), axis=1)

	print('S: Train Error', trainError(predictions['s'], testLabels['s']))
	print('W: Train Error', trainError(predictions['w'], testLabels['w']))
	print('K: Train Error', trainError(predictions['k'], testLabels['k']))
	print('All: Train Error', trainError(joined_predictions, labels))
