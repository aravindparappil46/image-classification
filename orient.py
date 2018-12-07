"""
------------------------------------------------------------------------------------------
Elements of AI | Assignment 4 | mkpandey-aparappi-atarfe
------------------------------------------------------------------------------------------
"""

import sys
import re
import math
from collections import defaultdict
import numpy as np
import adaboost as adaboost
from knn import *
from forest import *

# For testing purposes..remove below line for actual program
sys.argv = ['program_name','train','train-data.txt','model_file.txt', 'adaboost']

# Fetching cmd-line args
train_or_test = sys.argv[1]
input_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]
    
# Read input file into a variable..skips first column with image name
data = np.loadtxt(input_file, usecols=range(1,194))

# Based on the model required, call respective functions
if model == 'best':
    model = 'nearest'
    
if model == 'adaboost':
    adaboost.train(data)

elif model == 'nearest':
    print('K Nearest Neighbors model')

elif model == 'forest':
    print('Decision Trees')


