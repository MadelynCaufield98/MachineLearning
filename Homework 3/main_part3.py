import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression


# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: do not use the scikit-learn or other third-party modules for this step
maxIndex = len(X)
randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace=False)
trainX = X[randomTrainingSamples]
trainY = y[randomTrainingSamples]

randomTestingSamples = np.setdiff1d(np.arange(maxIndex), randomTrainingSamples)
testX = X[randomTestingSamples]
testY = y[randomTestingSamples]

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"
# use sklearn class
clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(trainX,trainY)
# scores over testing samples
print(clf.score(testX,testY))

# self-developed model
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

##implementation of sigmoid function
def Sigmoid(x):
	g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
	return g

##Prediction function
def Prediction(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)


# implementation of cost functions
def Cost_Function(X,y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		est_yi = Prediction(theta,xi)
		if y[i] == 1:
			error = y[i] * math.log(est_yi)
		elif y[i] == 0:
			error = (1-y[i]) * math.log(1-est_yi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	#print 'cost is ', J
	return J


# gradient components called by Gradient_Descent()

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = Prediction(theta,X[i])
		error = (hi - y[i])*xij
		sumErrors += error
	m = len(y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

# execute gradient updates over thetas
def Gradient_Descent(X,y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		deltaF = Cost_Function_Derivative(X,y,theta,j,m,alpha)
		new_theta_value = theta[j] - deltaF
		new_theta.append(new_theta_value)
	return new_theta

theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
if my_score > scikit_score:
	print('You won!')
elif my_score == scikit_score:
	print('Its a tie!')
else:
	print('Scikit won.. :(')
print('Your score: ', my_score)
print('Scikits score: ', scikit_score)
######################PLACEHOLDER2 #end #########################


# step 3: Use the model to get class labels of testing samples.

######################PLACEHOLDER3 #start#########################
# codes for making prediction,
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions
#xHat = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)  # add column of 1s to left most  ->  130 X 3
#negXHat = np.negative(xHat)  # -1 multiplied by matrix -> still 130 X 3
#hatProb = 1.0 / (1.0 + np.exp(negXHat * bHat))  # variant of classification   -> 130 X 3
# predict the class labels with a threshold
#yHat = (hatProb >= 0.5).astype(int)  # convert bool (True/False) to int (1/0)
import numpy as np
def gradientDescent(testX, testY, alpha = 0.01, max_iter = 500):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.001, 110000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()
#PLACEHOLDER#end

######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation
yHat = (1/1+np.exp(-testX@beta_hat) >= 0.5).astype(int)
testYDiff = np.abs(yHat - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

print('average error: {} ({})'.format(avgErr, stdErr))

######################PLACEHOLDER4 #start#########################
#Please add codes to calculate the ROC curves and AUC of the trained model over all the testing samples
# use sklearn class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
trainX = sc_X.fit_transform(trainX)
testX = sc_X.transform(testX)

from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(trainX, trainY)

y_pred_svm = model_SVC.decision_function(testX)
clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(trainX,trainY)

y_pred_logistic = clf.decision_function(testX)

from sklearn.metrics import roc_curve, auc

logistic_fpr, logistic_tpr, threshold = roc_curve(testY, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(testY, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % auc_svm)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()

# KNN
#alphas = [0.001, 0.002, 0.005, 0.01, 0.02,0.05, 0.07, 0.1,0.5, 1.0]

#Fold1: alpha = 0.001
trainX1 = trainX[0:95,:]
trainY1 = trainY[0:95,:]
testX1 = trainX[96:119,:]
testY1 = trainY[96:119,:]

theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

#Fold2: alpha = 0.001
trainX2 = trainX[24:119,:]
trainY2 = trainY[24:119,:]
testX2 = trainX[0:23,:]
testY2 = trainY[0:23,:]

theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

#Fold3: alpha = 0.001
trainxx = trainX[0:23,:]
trainxxx = trainX[:-48,:]
trainX3 = np.concatenate([trainxx,trainxxx])
test = trainY[0:23,:]
test2 = trainY[:-48,:]
trainY3 = np.concatenate([test,test2])
testX3 = trainX[24:47,:]
testY3 = trainY[24:47,:]

theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

#Fold4: alpha = 0.001
trainxx = trainX[0:47,:]
trainxxx = trainX[:-72,:]
trainX4 = np.concatenate([trainxx,trainxxx])
trainyy = trainY[0:47,:]
trainyyy = trainY[:-72,:]
trainY4 = np.concatenate([trainyy,trainyyy])
testX4 = trainX[48:71,:]
testY4 = trainY[48:71,:]

theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

#Fold5: alpha = 0.001
trainxx = trainX[0:71,:]
trainxxx = trainX[:-96,:]
trainX5 = np.concatenate([trainxx,trainxxx])
trainyy = trainY[0:71,:]
trainyyy = trainY[:-96,:]
trainY5 = np.concatenate([trainyy,trainyyy])
testX5 = trainX[72:95,:]
testY5 = trainY[72:95,:]

theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.001 is " + str(round(average, 2)))

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.002 is " + str(round(average, 2)))

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.005 is " + str(round(average, 2)))

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.01 is " + str(round(average, 2)))

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.02 is " + str(round(average, 2)))

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.05 is " + str(round(average, 2)))

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.07 is " + str(round(average, 2)))

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.1 is " + str(round(average, 2)))

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 0.5 is " + str(round(average, 2)))

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX1,testY1)
length = len(testX1)
for i in range(length):
	prediction = round(Prediction(testX1[i],theta))
	answer = testY1[i]
	if prediction == answer:
		score += 1

my_score1 = float(score) / float(length)
my_score1

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX2,testY2)
length = len(testX2)
for i in range(length):
	prediction = round(Prediction(testX2[i],theta))
	answer = testY2[i]
	if prediction == answer:
		score += 1

my_score2 = float(score) / float(length)
my_score2

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX3,testY3)
length = len(testX3)
for i in range(length):
	prediction = round(Prediction(testX3[i],theta))
	answer = testY3[i]
	if prediction == answer:
		score += 1

my_score3 = float(score) / float(length)
my_score3

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX4,testY4)
length = len(testX4)
for i in range(length):
	prediction = round(Prediction(testX4[i],theta))
	answer = testY4[i]
	if prediction == answer:
		score += 1

my_score4 = float(score) / float(length)
my_score4

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations

m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX5,testY5)
length = len(testX5)
for i in range(length):
	prediction = round(Prediction(testX5[i],theta))
	answer = testY5[i]
	if prediction == answer:
		score += 1

my_score5 = float(score) / float(length)
my_score5

scores = [my_score1,my_score2,my_score3,my_score4,my_score5]

average = sum(scores) / len(scores)

print("The average accuracy for alpha = 1 is " + str(round(average, 2)))

########################################################################

# alpha = 0.001
theta = [0,0] #initial model parameters
alpha = 0.001 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.001, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.002
theta = [0,0] #initial model parameters
alpha = 0.002 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.002, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.005
theta = [0,0] #initial model parameters
alpha = 0.005 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.005, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.01
theta = [0,0] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.01, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.02
theta = [0,0] #initial model parameters
alpha = 0.02 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.02, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.05
theta = [0,0] #initial model parameters
alpha = 0.05 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.05, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.07
theta = [0,0] #initial model parameters
alpha = 0.07 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.07, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.1
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.1, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 0.5
theta = [0,0] #initial model parameters
alpha = 0.5 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 0.5, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()

# alpha = 1
theta = [0,0] #initial model parameters
alpha = 1 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
	# call the functions for gradient descent method
	new_theta = Gradient_Descent(X,y,theta,m,alpha)
	theta = new_theta
	if x % 200 == 0:
		# calculate the cost function with the present theta
		Cost_Function(X,y,theta,m)
		print('theta ', theta)
		print('cost is ', Cost_Function(X,y,theta,m))

# comparing accuracies of two models.
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
	prediction = round(Prediction(testX[i],theta))
	answer = testY[i]
	if prediction == answer:
		score += 1

my_score = float(score) / float(length)
my_score

def gradientDescent(testX, testY, alpha, max_iter):
    beta = np.zeros(testX.shape[1])
    testY = testY.flatten()
    for it in range(max_iter):
        p=1/(1+np.exp(-testX @ beta))
        score = (testY-p) @ testX
        beta = beta + (alpha/testY.size)*score
    return beta

beta_hat = gradientDescent(testX, testY, 1, 1000)
beta_hat

testY = testY.flatten()
pd.crosstab(testY, (1/(1+np.exp(-testX@beta_hat))>=0.5)+0)

from statsmodels.discrete.discrete_model import Logit
testY = testY.flatten()
Logit_model = Logit(testY, testX).fit()
Logit_model.pred_table()
