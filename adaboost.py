"""
AdaBoost method
Author: Aravind Parappil
"""
import numpy as np
import math
import pickle as pickle
from collections import defaultdict
import operator

def train(data, train_orientation, random_hyp_pairs, max_iterations, hyp_alphas):
    print("Training for ", train_orientation)
    weights = [1/float(len(data))]* len(data)
    my_labels = []
       
    for i in range(0, max_iterations):              
        column_1 = random_hyp_pairs[i][0]
        column_2 = random_hyp_pairs[i][1]
        
        error = 0.0
        # Assigning labels based on hyp    
        for j in range(len(data)):
            if data[j][column_1] - data[j][column_2] > 0:
                my_labels.append([train_orientation])
            else:
                my_labels.append([90, 180, 270])

            # First col of data contains actual labels. Comparing....
            # if mismatch, then update error as the sum of weight at row mismatched
            
            if  data[:,0][j] not in my_labels[j]:      
                error += weights[j]
                
        
        # Error value has been calculated now...
        # Updating weights for correctly classified rows
        
        print(error)
        if error < 0.5: 
            for k in range(len(data)):
                if data[:,0][k] in my_labels[k]:
                    weights[k] = weights[k] * (float(error)/(1-error))

            #weights = normalize(weights)
            total_sum = sum(weights)
            weights = [i/total_sum for i in weights]
            hyp_alphas[train_orientation][(column_1, column_2)] = math.log(float((1-error)/error))
            
    return hyp_alphas


# Divide each element with sum of array
def normalize(weights):
    total_sum = weights.sum()
    return np.divide(weights, total_sum)


# Testing
def test(data, hyp_alphas, image_names):
    f = open('output.txt', 'a')
    correct_prediction_count = 0
    
    for file_index in range(len(data)):
        predicted_orientation_dict = {}
        for k,v in hyp_alphas.items():
            summed_up = 0
            for hyp_tuple, alpha in v.items():
                diff = data[file_index][hyp_tuple[0]] - data[file_index][hyp_tuple[1]]
                if diff > 0:
                    prod = 1 * alpha
                else:
                    prod = -1 * alpha
                summed_up += prod
            predicted_orientation_dict[k]= summed_up
        
        # Finding which orientation has the largest value
        # REFERENCE: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        final_prediction = str(max(predicted_orientation_dict.items(), key=operator.itemgetter(1))[0])
        f.write(image_names[file_index]+' '+ final_prediction+'\n')

        actual_orient = data[file_index][1]
        if int(final_prediction) == actual_orient:
            correct_prediction_count += 1
    
    # Finding how many records were correctly identified
    print('Accuracy: ',(correct_prediction_count/len(data)))
