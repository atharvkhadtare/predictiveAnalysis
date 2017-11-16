# %% 1 
# Package imports 
from keras.models import Sequential, h5py, load_model
import matplotlib
import numpy
import h5py
from keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler, normalize
 # load pima indians dataset

epochsPerLoop = 500
epochLoops = 3
fittingVerbose = 0

datasetLearning = numpy.loadtxt("learning_set/nnDataSet2_2.csv", delimiter=",")

TestSetFilenameCollection = ["learning_set/nnDataSet1_1.csv",
							 "learning_set/nnDataSet1_2.csv",
							 "learning_set/nnDataSet2_1.csv",
							 "learning_set/nnDataSet3_1.csv",
							 "learning_set/nnDataSet3_2.csv",
							 "test_set/nnDataSet1_3.csv",
							 "test_set/nnDataSet1_4.csv",
							 "test_set/nnDataSet1_5.csv",
							 "test_set/nnDataSet1_6.csv",
							 "test_set/nnDataSet1_7.csv",
							 "test_set/nnDataSet2_3.csv",
							 "test_set/nnDataSet2_4.csv",
							 "test_set/nnDataSet2_5.csv",
							 "test_set/nnDataSet2_6.csv",
							 "test_set/nnDataSet2_7.csv",
							 "test_set/nnDataSet3_3.csv",]




##########################################################################################################################
def PredictForTestSetCollection(filenames):
	for filename in filenames:
		predictForTestSet(filename)

def predictForTestSet(filename):
	datasetPredict = numpy.loadtxt(filename, delimiter=",")

	totalSizePredict = len(datasetPredict)-2
	# print(totalSizePredict)
	# numpy.random.shuffle(dataset)
	XPredict = numpy.zeros((totalSizePredict,6))
	for j in range(totalSizePredict):
		XPredict[j][0] = datasetPredict[j,0]
		XPredict[j][2] = datasetPredict[j,1]
		XPredict[j][4] = datasetPredict[j,2]

	for j in range(1,totalSizePredict):
		XPredict[j][1] = XPredict[j-1][0]
		XPredict[j][3] = XPredict[j-1][2]
		XPredict[j][5] = XPredict[j-1][4]
	YPredict = datasetPredict[:,3]
	# print (X)
	# print (Y)

	MaxPredict = numpy.zeros(len(XPredict[0]))
	XPredict = XPredict.astype(float)
	for i in range(len(XPredict[0])):
		MaxPredict[i] = 0
		for j in range(len(XPredict)):
			if(MaxPredict[i] < XPredict[j][i]):
				MaxPredict[i] = XPredict[j][i]
		# print ("Max = ", MaxPredict[i])
		for j in range(len(XPredict)):
			XPredict[j][i] = XPredict[j][i] / MaxPredict[i]

	YPredict = YPredict.astype(float)

	MaxYPredict = 0
	for i in range(len(YPredict)):
		if(MaxYPredict < YPredict[i]):
			MaxYPredict = YPredict[i]
	# print ("MaxYPredict = ", MaxYPredict)
	for i in range(len(YPredict)):
		YPredict[i] = YPredict[i] / MaxYPredict

	errorSumPredict=0
	totalReadingsPredict = 0
	for j in range((totalSizePredict)):
		a = model.predict(numpy.array(XPredict[j]).reshape(-1,6))
		if(YPredict[j] == 0):
			continue
		error_percPredict = abs(((YPredict[j]-a)*100)/YPredict[j])
		# print("error_perc [", j, "] = ", error_percPredict)
		errorSumPredict = errorSumPredict+error_percPredict
		totalReadingsPredict += 1
	print ("For Test Set - ", filename, ", Avg Error in Test Data Set = ", (errorSumPredict/(totalReadingsPredict)))



##########################################################################################################################




def createModel():
	model = Sequential()
	model.add(Dense(13, input_dim=6, init = 'normal', activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(10,init = 'normal', activation='relu'))
	# model.add(Dense(3,  init = 'normal', activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1, init = 'normal'))
	# model.add(Dense(1, init = 'normal',  activation='sigmoid'))


	# Compile model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
	return model



##########################################################################################################################



model = createModel()

totalSize = len(datasetLearning)-2
# print(totalSize)
trainLimit = int(((len(datasetLearning)-2)/100)*100)
# print(trainLimit)

numpy.random.shuffle(datasetLearning)

X = numpy.zeros((totalSize,6))
for j in range(totalSize):
	X[j][0] = datasetLearning[j,0]
	X[j][2] = datasetLearning[j,1]
	X[j][4] = datasetLearning[j,2]
for j in range(1,totalSize):
	X[j][1] = X[j-1][0]
	X[j][3] = X[j-1][2]
	X[j][5] = X[j-1][4]
Y = datasetLearning[:,3]
# print (X)
# print (Y)

Max = numpy.zeros(len(X[0]))
X = X.astype(float)
for i in range(len(X[0])):
	Max[i] = 0
	for j in range(len(X)):
		if(Max[i] < X[j][i]):
			Max[i] = X[j][i]
	# print ("Max = ", Max[i])
	for j in range(len(X)):
		X[j][i] = X[j][i] / Max[i]

Y = Y.astype(float)

MaxY = 0
for i in range(len(Y)):
	if(MaxY < Y[i]):
		MaxY = Y[i]
# print ("MaxY = ", MaxY)
for i in range(len(Y)):
	Y[i] = Y[i] / MaxY

# print (X)
# print (Y)
# create model


# Fit the model
print(X.shape)
for i in range(epochLoops):
	model.fit(X[:trainLimit], Y[:trainLimit], epochs=epochsPerLoop, verbose=fittingVerbose, validation_split=0.2, batch_size=10)
	model.save('3_nn13_10_mean_squared_error_sgd_bs10_split02_epochs_'+str(i)+'0000.h5')
	scores=model.evaluate(X[:trainLimit],Y[:trainLimit],batch_size=10)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


	# for j in  range(10):
	# 	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
	# 	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
	# 	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
	# 	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

	errorSum=0
	totalReadings = 0
	for j in range((totalSize-1)):
		a = model.predict(numpy.array(X[j]).reshape(-1,6))
		if(Y[j] == 0):
			continue
		error_perc = abs(((Y[j]-a)*100)/Y[j])
		errorSum = errorSum+error_perc
		totalReadings += 1
	print("\n\nAfter ", (i+1)*epochsPerLoop, "epochs,")
	print ("Avg Error in Learning Data Set = ", (errorSum/(totalReadings)))
	# predictForTestSet(TestSetFilename)
	PredictForTestSetCollection(TestSetFilenameCollection)



# print (model)
# a = model.predict(numpy.array([0.00125628, 0, 0.07505039, 0.07033415, 0.06912259, 0.06125143]).reshape(-1,6))
# for i in  range(10):
# 	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
# 	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
# 	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
# 	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

# print (a)
# evaluate the model
scores = model.evaluate(X[:trainLimit], Y[:trainLimit])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))