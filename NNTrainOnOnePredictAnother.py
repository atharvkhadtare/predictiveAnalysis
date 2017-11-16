# %% 1 
# Package imports 
from keras.models import Sequential, h5py, load_model
import matplotlib
import numpy
import h5py
from keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler, normalize
 # load pima indians dataset

epochsPerLoop = 1000
epochLoops = 3

datasetPredict = numpy.loadtxt("nnDataSet3_1.csv", delimiter=",")

dataset = numpy.loadtxt("nnDataSet.csv", delimiter=",")


totalSize = len(dataset)-2
print(totalSize)
# totalSize = 795
trainLimit = int(((len(dataset)-2)/100)*50)
print(trainLimit)
# trainLimit = 400
# dataset.shuffle()
numpy.random.shuffle(dataset)
# # split into input (X) and output (Y) variables
# X = dataset[:,0:3]
X = numpy.zeros((totalSize,6))
# X[:,0] = dataset[:,0]
# X[2] = dataset[:,1]
# X[4] = dataset[:,2]
# for i in range(0,2):
for j in range(totalSize):
	X[j][0] = dataset[j,0]
	X[j][2] = dataset[j,1]
	X[j][4] = dataset[j,2]
	# X[j][i*2] = dataset[j,i]
# for i in (1,3,5):
# 	X[0][i] = X[0][i-1]
# 	for j in range(1,len(X[:][0]-1)):
# 		X[j][i] = X[j-1][i-1]
# for i in (1,3,5):
# 	X[0][i] = X[0][i-1]
for j in range(1,totalSize):
	X[j][1] = X[j-1][0]
	X[j][3] = X[j-1][2]
	X[j][5] = X[j-1][4]
Y = dataset[:,3]
# print (X)
# print (Y)

# y = numpy.random.randint(0,10,size=(300,5))
# y = (y*10)
# print (y)
Max = numpy.zeros(len(X[0]))
X = X.astype(float)
for i in range(len(X[0])):
	Max[i] = 0
	for j in range(len(X)):
		if(Max[i] < X[j][i]):
			Max[i] = X[j][i]
	print ("Max = ", Max[i])
	for j in range(len(X)):
		X[j][i] = X[j][i] / Max[i]

Y = Y.astype(float)

MaxY = 0
for i in range(len(Y)):
	if(MaxY < Y[i]):
		MaxY = Y[i]
print ("MaxY = ", MaxY)
for i in range(len(Y)):
	Y[i] = Y[i] / MaxY

# for i in range(len(X)):
# 	sum = 0
# 	for j in range(len(X[i])-1):
# 		sum = sum + X[j][i]
# 	X[i][len(X[i])-1]= sum/(len(X[i])-1)
# 	print (X[i][len(X[i])-1])
# print (X)

# X = y[:,0:4]
# Y = y[:,4]
# X[:][0] = X[:][0] / numpy.linalg.norm(X[:][0])
# sc = StandardScaler()
# X = sc.fit_transform(X)
# for i in range(len(Y)):
# 	Y[i] = Y[i]/10
# X = X/10
# Y = sc.fit_transform(Y)
# Y= Y.reshape(1,-1)
# print (X[4][10])
# print("max X = ", max(X[:][0]))
# print("max X = ", max(X[:][1]))
# print("max X = ", max(X[:][2]))
# print("len(X[:]) = ", len(X[:]))
# for i in range(len(X[0])):
# 	for j in range(len(X[:]))
# 		X[j][i] = X[j][i]/max(X[:][i])

print (X)
print (Y)
print (X.dtype)
print (Y.dtype)
# create model









##########################################################################################################################




totalSizePredict = len(datasetPredict)-2
print(totalSizePredict)
# totalSize = 795
# trainLimit = 400
# dataset.shuffle()
# numpy.random.shuffle(dataset)
# # split into input (X) and output (Y) variables
# X = dataset[:,0:3]
XPredict = numpy.zeros((totalSizePredict,6))
# X[:,0] = dataset[:,0]
# X[2] = dataset[:,1]
# X[4] = dataset[:,2]
# for i in range(0,2):
for j in range(totalSizePredict):
	XPredict[j][0] = datasetPredict[j,0]
	XPredict[j][2] = datasetPredict[j,1]
	XPredict[j][4] = datasetPredict[j,2]
	# X[j][i*2] = dataset[j,i]
# for i in (1,3,5):
# 	X[0][i] = X[0][i-1]
# 	for j in range(1,len(X[:][0]-1)):
# 		X[j][i] = X[j-1][i-1]
# for i in (1,3,5):
# 	X[0][i] = X[0][i-1]
for j in range(1,totalSizePredict):
	XPredict[j][1] = XPredict[j-1][0]
	XPredict[j][3] = XPredict[j-1][2]
	XPredict[j][5] = XPredict[j-1][4]
YPredict = datasetPredict[:,3]
# print (X)
# print (Y)

# y = numpy.random.randint(0,10,size=(300,5))
# y = (y*10)
# print (y)
MaxPredict = numpy.zeros(len(XPredict[0]))
XPredict = XPredict.astype(float)
for i in range(len(XPredict[0])):
	MaxPredict[i] = 0
	for j in range(len(XPredict)):
		if(MaxPredict[i] < XPredict[j][i]):
			MaxPredict[i] = XPredict[j][i]
	print ("Max = ", MaxPredict[i])
	for j in range(len(XPredict)):
		XPredict[j][i] = XPredict[j][i] / MaxPredict[i]

YPredict = YPredict.astype(float)

MaxYPredict = 0
for i in range(len(YPredict)):
	if(MaxYPredict < YPredict[i]):
		MaxYPredict = YPredict[i]
print ("MaxYPredict = ", MaxYPredict)
for i in range(len(YPredict)):
	YPredict[i] = YPredict[i] / MaxYPredict




def predictOnNewSet():
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
	print ("Avg Error in New Data Set  = ", (errorSumPredict/(totalReadingsPredict)))
	print ("Avg Error in Same Data Set = ", (errorSum/(totalReadings)))


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

model = createModel()

# Fit the model
print(X.shape)
for i in range(epochLoops):
	model.fit(X[:trainLimit], Y[:trainLimit], epochs=epochsPerLoop, validation_split=0.2, batch_size=10)
	model.save('3_nn13_10_mean_squared_error_sgd_bs10_split02_epochs_'+str(i)+'0000.h5')
	scores=model.evaluate(X[:trainLimit],Y[:trainLimit],batch_size=10)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	for j in  range(10):
		rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
		a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
		error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
		print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)
	totalReadings = 0
	errorSum=0

	for j in range(totalSize-trainLimit-1):
		a = model.predict(numpy.array(X[j]).reshape(-1,6))
		if(Y[j] == 0):
			continue
		error_perc = abs(((Y[j]-a)*100)/Y[j])
		errorSum = errorSum+error_perc
		totalReadings += 1

	predictOnNewSet()



# print (model)
# a = model.predict(numpy.array([0.00125628, 0, 0.07505039, 0.07033415, 0.06912259, 0.06125143]).reshape(-1,6))
for i in  range(10):
	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

errorSum=0
totalReadings = 0
for j in range((totalSize-trainLimit-1)):
	a = model.predict(numpy.array(X[j]).reshape(-1,6))
	if(Y[j] == 0):
		continue
	error_perc = abs(((Y[j]-a)*100)/Y[j])
	errorSum = errorSum+error_perc
	totalReadings += 1
print ("Avg Error = ", (errorSum/(totalReadings)))

predictOnNewSet()
# print (a)
# evaluate the model
scores = model.evaluate(X[:trainLimit], Y[:trainLimit])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))