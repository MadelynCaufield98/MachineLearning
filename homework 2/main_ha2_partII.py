from sklearn.linear_model import LinearRegression
import numpy as np
from download_data import download_data
from sklearn.preprocessing import MinMaxScaler
from dataNormalization import rescaleMatrix

# calling data and rescaling
sat = download_data('sat.csv', [1, 2, 4]).values
sat = np.array(rescaleMatrix(sat))

# splitting features and labels
X = sat[:,0:1]
y = sat[:,2]

# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

theta = np.zeros(3)

xValues = np.ones((60,3))
xValues[:,1:3]=satTrain[:,0:2]
yValues = satTrain[:, 2]

testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

# Linear Regression
lr = LinearRegression().fit(xValues,yValues)
print("Training set score: {:.2f}".format(lr.score(xValues,yValues)))
print("Test set score: {:.2f}".format(lr.score(testXValues,testYValues)))

# R^2 for linear regression
yHat = lr.predict(testXValues)

Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - yHat)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

# Ridge regression
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=5).fit(xValues,yValues)
print("Training set score: {:.2f}".format(ridge.score(xValues,yValues)))
print("Test set score: {:.2f}".format(ridge.score(testXValues,testYValues)))

# R^2 for ridge regression
yHat = ridge.predict(testXValues)

Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - yHat)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

# LASSO
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0001,max_iter=100000).fit(xValues,yValues)
print("Training set score: {:.2f}".format(lasso.score(xValues,yValues)))
print("Test set score: {:.2f}".format(lasso.score(testXValues,testYValues)))
print("Number of features used:",np.sum(lasso.coef_ != 0))

# R^2 for LASSO
yHat = lasso.predict(testXValues)

Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - yHat)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

# Elastic net
X = sat[:,0:1]
y = sat[:,2]

# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

trainX = satTrain[:,0:1]
trainY = satTrain[:,2]
testX = satTest[:,0:1]
testY = satTest[:,2]

from sklearn.linear_model import ElasticNetCV

ENet = ElasticNetCV(cv=5,random_state=0)
ENet.fit(X,y)
print("Training R^2: {:.2f}".format(ENet.score(trainX,trainY)))
print("Test R^2: {:.2f}".format(ENet.score(testX,testY)))












import os
os.getcwd()

os.chdir('/Users/madelyncaufield/Desktop/ECON425 Machine Learning 1/homework 2/part1')
