from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradientDescent
from dataNormalization import rescaleMatrix


#NOTICE: Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# There are two PLACEHODERS IN THIS SCRIPT

# parameters

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves.
ALPHA = 0.1
MAX_ITER = 500
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
sat = rescaleMatrix(sat)
################PLACEHOLDER2 #end##########################


# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

##############################################################################

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves.
ALPHA = 0.001
MAX_ITER = 10000
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
sat = rescaleMatrix(sat)
################PLACEHOLDER2 #end##########################


# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

##############################################################################

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their nconvergence curves.
ALPHA = 0.05
MAX_ITER = 500
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
sat = rescaleMatrix(sat)
################PLACEHOLDER2 #end##########################


# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

##############################################################################

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves.
ALPHA = 0.01
MAX_ITER = 500
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
sat = rescaleMatrix(sat)
################PLACEHOLDER2 #end##########################


# training data;
satTrain = sat[0:60, :]
# testing data;
satTest = sat[60:len(sat),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

##############################################################################

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves.
ALPHA = 0.01
MAX_ITER = 1000
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
from dataNormalization import rescaleNormalization
sat_norm = rescaleNormalization(sat)
#################PLACEHOLDER2 #end##########################


# training data;
sat_norm = np.array(sat_norm)
satTrain = sat_norm[0:60, :]
# testing data;
satTest = sat_norm[60:len(sat_norm),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)

##############################################################################

################PLACEHOLDER1 #start##########################
# test multiple learning rates and report their convergence curves.
ALPHA = 0.01
MAX_ITER = 1000000
################PLACEHOLDER1 #end##########################

#% step-1: load data and divide it into two subsets, used for training and testing
sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

################PLACEHOLDER2 #start##########################
# Normalize data
from dataNormalization import meanNormalization
sat_mean = meanNormalization(sat)
#################PLACEHOLDER2 #end##########################


# training data;
sat_mean = np.array(sat_mean)
satTrain = sat_mean[0:60, :]
# testing data;
satTest = sat_mean[60:len(sat_mean),:]

#% step-2: train a linear regression model using the Gradient Descent (GD) method
# ** theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
theta = np.zeros(3)

xValues = np.ones((60, 3))
xValues[:, 1:3] = satTrain[:, 0:2]
yValues = satTrain[:, 2]
# call the GD algorithm, placeholders in the function gradientDescent()
[theta, arrCost] = gradientDescent(xValues, yValues, theta, ALPHA, MAX_ITER)


#visualize the convergence curve
plt.plot(range(0,len(arrCost)),arrCost);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

#% step-3: testing
testXValues = np.ones((len(satTest), 3))
testXValues[:, 1:3] = satTest[:, 0:2]
tVal =  testXValues.dot(theta)
testYValues = satTest[:,2]

#% step-4: evaluation
# calculate average error and standard deviation
tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
print('results: {} ({})'.format(np.mean(tError), np.std(tError)))
################PLACEHOLDER3 #start##########################
# Calculate and print R2
Ybar = np.mean(testYValues)
numerator = np.sum( (satTest[:,2] - tVal)**2 )
denominator = np.sum( (satTest[:,2] - Ybar)**2 )

sat_R2 = float(1 - (numerator/denominator))
print("Out of Sample R^2: ", sat_R2)











import os
os.getcwd()

os.chdir('/Users/madelyncaufield/Desktop/ECON425 Machine Learning 1/homework 2/part1')
