"""
------------------------------------------------------------------------------------------
Elements of AI | Assignment 4 | mkpandey-aparappi-atarfe
------------------------------------------------------------------------------------------
"""

import sys
import re
import math
from collections import defaultdict

# For testing purposes..remove below line for actual program
sys.argv = ['program_name','train','train_file.txt','model_file.txt', 'adaboost']

# Fetching cmd-line args
model = sys.argv[4]
train_or_test = sys.argv[1]
input_file = sys.argv[2]
model_file = sys.argv[3]

# Read file into a variable
with open(input_file, 'r', encoding="utf-8") as file:
        raw_input = file.read()
