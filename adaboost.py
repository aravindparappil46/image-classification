#!/usr/bin/env python3

"""
AdaBoost method
Author: Aravind Parappil
Discussed with Shivam Thakur for training logic
"""
import numpy as np
import math
import pickle as pickle
from collections import defaultdict
import operator
import random

# Training
def train(data, train_orientation, random_hyp_pairs, max_iterations, hyp_alphas):
    print("Training for ", train_orientation)
    
    weights = [1/float(len(data))]* len(data)
    al = [0,90,180,270]
    others=[element for element in al if element != train_orientation]

    # Iterates till a threshold
    # Implementation of calculating error was referred from Shivam Thakur
    for i in range(0, max_iterations):              
        column_1 = data[:,random_hyp_pairs[i][0]]
        column_2 = data[:,random_hyp_pairs[i][1]]
        hypothesis = column_1 - column_2
        my_labels = []

        # Assigning labels based on hyp 
        for hyp in hypothesis:
            if hyp > 0:
                my_labels.append([train_orientation])
            else:
                my_labels.append(others)
                
        error = 0.0
         
        for j in range(len(data)):
            # First col of data contains actual labels. Comparing....
            # if mismatch, then update error as the sum of weight at row mismatched
            if data[:,0][j] not in my_labels[j]:
                error += weights[j]

        # Error value has been calculated now...
        # Updating weights for correctly classified rows
        # Does this only if error < (1-error) to avoid Math errors while log op
        
        if error < (1-error):
            for k in range(len(data)):
                if data[:,0][k] in my_labels[k]:
                    weights[k] = weights[k] * (float(error)/(1-error))

            # Normalizing weights
            weights = normalize(weights)
    
            hyp_alphas[train_orientation][(random_hyp_pairs[i][0], \
                                           random_hyp_pairs[i][1])] = \
                                           math.log(float((1-error)/error))          
    return hyp_alphas


# Divide each element with sum of array
# to normalize
def normalize(weights):
    total = sum(weights)
    weights = [i/total for i in weights]
    return weights


# Testing
def test(data, hyp_alphas, image_names):
    f = open('output.txt', 'a')
    correct_prediction_count = 0

    # Dictionary that contains a1h1+a2h2+...anhn values for each weak learner
    predicted_orientation_dict = {0:[], 90: [], 180: [], 270: []}
    
    for k,v in hyp_alphas.items():
        summed_up = [0.0]* len(data)
        for hyp_tuple, alpha in v.items():
            column_1 = data[:,hyp_tuple[0]]
            column_2 = data[:,hyp_tuple[1]]
            diff = column_1 - column_2            
            prod = []
            for value in diff:
                if value > 0:
                    prod.append(1 * alpha)
                else:
                    prod.append(-1 * alpha)
                    
            for i in range(len(diff)):
                summed_up[i] += prod[i]
                
        predicted_orientation_dict[k].append(summed_up)

    # All predicted values in hand now...
    # Checking for best value
    predict_labels(data, image_names, predicted_orientation_dict)


# Checks which learner's value was > 0 for each image
# and assigns that as the classification label. If conflict, takes the
# biggest value among these.
# If none of the values are a positive number, randomly assigns 
# an orientation
def predict_labels(data, image_names, predicted_orientation_dict):
    correct_prediction_count = 0
    f = open('output.txt', 'a')
    
    for img_index in range(len(data)):
        best_orient = []
        for curr_orient, v in predicted_orientation_dict.items():
            if np.sign(v[0][img_index]) == 1:
                if len(best_orient) == 0:
                    best_orient.append(curr_orient)
                    curr_val = v[0][img_index]
                else:
                    if v[0][img_index] > curr_val:
                        curr_val = v[0][img_index]
                        best_orient = []
                        best_orient.append(curr_orient)

            # All learners predicted negative values..randomly assign a label
            else:
                best_orient.append(random.choice([0,90,180,270]))

        # Writing to output.txt and counting if classification was correct or not   
        f.write(image_names[img_index]+' '+ str(best_orient[0])+'\n')
        if(best_orient[0] == data[img_index][0]):
            correct_prediction_count += 1
    
    print('Accuracy: ',100*correct_prediction_count/len(data))
