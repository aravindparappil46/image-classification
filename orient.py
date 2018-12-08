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
from knn import *
from forest import *

# For testing purposes..remove below line for actual program
sys.argv = ['program_name','test','test-data.txt','model-file.pickle', 'adaboost']

# Fetching cmd-line args
train_or_test = sys.argv[1]
input_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]
    
# Read input file into a variable..skips first column with image name
data = np.loadtxt(input_file, usecols=range(1,194))

# All image names stored in separate array
image_names = np.genfromtxt(input_file, dtype="str", usecols=range(0,1))

# Based on the model required, call respective functions
if model == 'best':
    model = 'nearest'
    
if model == 'adaboost':

    #----------------------------------#
    #            TRAINING              #
    #----------------------------------#
    if train_or_test == 'train':
        random_hyp_pairs = []
        hyp_alphas = defaultdict(dict)
        max_iterations = 20
        f = open(model_file, 'wb')

        # Finding total number of cols in data...Not counting the col with actual labels
        # Since np array lengths are calculated from index 1, have to decrement 1 too
        total_num_of_cols = len(data[0])-2
        
        for i in range(0,max_iterations):
            # Getting two random column indices. Will find difference between row vals
            # of these two columns, which will be our hypothesis while training
            # Idea credit: Shivam Thakur
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
        
        alphas_for_0 = adaboost.train(data, 0, random_hyp_pairs, max_iterations, f, hyp_alphas)
        alphas_for_90 = adaboost.train(data, 90, random_hyp_pairs, max_iterations, f, alphas_for_0)
        alphas_for_180 = adaboost.train(data, 180, random_hyp_pairs, max_iterations, f, alphas_for_90)
        alphas_for_270 = adaboost.train(data, 270, random_hyp_pairs, max_iterations, f, alphas_for_180)

        # Storing model params in a pickle to retain dictionary structure
        pickle.dump(alphas_for_270, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Training ends....
              
    #----------------------------------#
    #            TESTING               #
    #----------------------------------#
    elif train_or_test == 'test':      
        f = open(model_file, 'rb')
        hyp_alphas = pickle.load(f)
        adaboost.test(data, hyp_alphas, image_names)
        
        
    
elif model == 'nearest':
    print('K Nearest Neighbors model')

elif model == 'forest':
    print('Decision Trees')


