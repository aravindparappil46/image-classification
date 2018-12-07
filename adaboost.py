"""
AdaBoost method
Author: Aravind Parappil
"""
import numpy as np
import math
from collections import defaultdict
import random

hyp_alphas = defaultdict()
max_iterations = 20

def train(data):
    weights = np.array([[float(1/len(data))]]* len(data))
    my_labels = np.array([[-1]]* len(data))

    # Finding total number of cols in data...Not counting the col with actual labels
    # Since np array lengths are calculated from index 1, have to decrement 1 too
    total_num_of_cols = len(data[0])-2
    
    for i in range(0,max_iterations):
        error = 0
        column_1 = random.randint(1,total_num_of_cols)
        column_2 = random.randint(1,total_num_of_cols)

        # Resolving conflict, if any
        if column_1 == column_2:
            if column_2 != total_num_of_cols:
                column_2 += 1
            else:
                column_2 -= 1

        # Assigning labels based on hyp    
        for j in range(len(data)):
            if data[j][column_1] - data[j][column_2] < 0:
                my_labels[j] = 0
            else:
                my_labels[j] = -1

            # first col of data contains actual labels. Comparing....
            # if mismatch, then update error as the sum of weight for row mismatched
            if my_labels[j] !=  data[:,0][j]:
                error += weights[j]

        # Error value has been calculated now...
        # Updating weights for correctly classified rows 
        for k in range(len(data)):
            if my_labels[k] == data[:,0][k]:
                weights[k] *= (error/(1-error))

        weights = normalize(weights)
        hyp_alphas[(column_1, column_2)] = math.log((1-error)/error)

# Divide each element with sum of array
def normalize(weights):
    total_sum = weights.sum()
    return np.divide(weights, total_sum)
        
            

