import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for interation in range(0, numIterations):
        ################PLACEHOLDER4 #start##########################
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
	# Replace the following variables if needed
        gradient =  np.dot(transposedX, loss) / m
        theta = theta - alpha * gradient  # or theta = theta - alpha * gradient
        ################PLACEHOLDER4 #end##########################

        ################PLACEHOLDER5 #start##########################
        # calculate the current cost with the new theta;
        atmp =  np.sum(loss ** 2) / m
        print(atmp)
        arrCost.append(atmp)

        ################PLACEHOLDER5 #start##########################

    return theta, arrCost
