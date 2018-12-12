#!/usr/bin/env python3
"""
------------------------------------------------------------------------------------------
Elements of AI | Assignment 4 | mkpandey-aparappi-atarfe
------------------------------------------------------------------------------------------
"""

import sys
import re
import math
from collections import defaultdict
import random
import pickle as pickle
import numpy as np
import adaboost as adaboost
import knn as knn
import forest as forest
import datetime
import operator



# Fetching cmd-line args
train_or_test = sys.argv[1]
input_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]


# Based on the model required, call respective functions
if model == 'best':
    model = 'nearest'
    
if model == 'adaboost':
    
    # Read input file into a variable..skips first column with image name
    data = np.loadtxt(input_file, usecols=range(1,194))

    # All image names stored in separate array
    image_names = np.genfromtxt(input_file, dtype="str", usecols=range(0,1))
    #----------------------------------#
    #            TRAINING              #
    #----------------------------------#
    if train_or_test == 'train':
        random_hyp_pairs = []
        hyp_alphas = defaultdict(dict)
        max_iterations = 1227
        f = open(model_file, 'wb')

        start = datetime.datetime.now()
        
        # Finding total number of cols in data...Not counting the col with actual labels
        # Since np array lengths are calculated from index 1, have to decrement 1 too
        total_num_of_cols = len(data[0])-2

        # Generating random pairs of columns to compare
        print("Generating",max_iterations,"random pairs of column indices...")
        for i in range(0,max_iterations):
            # Getting two random column indices. Will find difference between row vals
            # of these two columns, which will be our hypothesis while training
            column_1 = random.randint(1,total_num_of_cols)
            column_2 = random.randint(1,total_num_of_cols)

            # Resolving conflict, if any. Shouldn't be the same
            if column_1 == column_2:
                if column_2 != total_num_of_cols:
                    column_2 += 1
                else:
                    column_2 -= 1
                    
            random_hyp_pairs.append((column_1,column_2))

        # Running train for the different orientations...
        # Each run will return a dict of dicts with key as particular orientation
        # Pipeline the dicts returned from one training stage to another till we finish.
        # Output will be a dict of dict with 4 keys (0, 90, 180, 270)
        
        print("Max iterations set to: ", max_iterations)
        alphas_for_0 = adaboost.train(data, 0, random_hyp_pairs, max_iterations, hyp_alphas)
        alphas_for_90 = adaboost.train(data, 90, random_hyp_pairs, max_iterations, alphas_for_0)
        alphas_for_180 = adaboost.train(data, 180, random_hyp_pairs, max_iterations, alphas_for_90)
        alphas_for_270 = adaboost.train(data, 270, random_hyp_pairs, max_iterations, alphas_for_180)
        
        # Storing model params in a pickle to retain dictionary structure
        pickle.dump(alphas_for_270, f, protocol=pickle.HIGHEST_PROTOCOL)
        end = datetime.datetime.now()

        print("Finished training in", end-start)
        # Training ends....
        f.close()
        
    #----------------------------------#
    #            TESTING               #
    #----------------------------------#
    elif train_or_test == 'test':      
        f = open(model_file, 'rb')
        # Load the model file and pass to test function
        hyp_alphas = pickle.load(f)
        adaboost.test(data, hyp_alphas, image_names)
        f.close()
        
    
elif model == 'nearest':
    
    file = open(input_file, "r")
    if train_or_test == 'train':
        train_data,train_names=knn.read_file(file)
        f = open(model_file, 'wb')
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    else:
        f = open(model_file, 'rb')
        train_data = pickle.load(f)
        test_data,test_names=knn.read_file(file)	
        knn.accuracy(knn.knn(knn.euclidean(train_data,test_data), train_data), test_names)
        f.close()

elif model == 'forest':

    if train_or_test == 'train':
       forest.train(input_file, model_file)

    else:
        
        forest.test_forest(input_file, model_file)

                

