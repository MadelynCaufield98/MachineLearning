# Question 1

# accuracy
7/20

# precision: dog
2/8

# precision: cat
1/5

# precision: monkey
3/7

# recall rate: dog
2/8

# recall rate: cat
1/6

# recall rate: monkey
3/6

# Question 2
def func_calConfusionMatrix(predY, trueY):

    #creating a matrix of zeroes of size 2 x 2
    K = len(np.unique(trueY))
    result = np.zeros((K, K))

    #calculate the values in the confusion matrix
    for i in range(len(true)):
        result[trueY[i]][predY[i]] += 1

    #storing the four values into four different variables, namely tp, tn, fp, fn
    tn = result[0][0]
    fp = result[0][1]
    fn = result[1][0]
    tp = result[1][1]

    accuracy = ((tn + tp)/(tn + tp + fn + fp))*100 #accuracy
    precision_postive = (tp/(fp + tp))*100 #precision of positive class
    precision_negative = (tn/(tn + fn))*100 #precision of negative class
    recall_positive = (tp/(fn + tp))*100 #recall of positive class
    recall_negative = (tn/(tn + fp))*100 #recall of negative class

    #return the values
    return accuracy, precision_postive, precision_negative, recall_positive, recall_negative
