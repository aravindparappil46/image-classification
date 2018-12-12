#!/usr/bin/env python3

# REFERENCE: Google Developers Youtube video https://www.youtube.com/watch?v=LDRbO9a6XPU

import math
import operator
import random
import pickle as pickle

# Node class to build the tree

class Node:
    depth = None
    column = None
    true = None
    false = None
    prediction = None

# Read the input file and store the image data
def read(path):
    file = open(path)
    images = list()
    predicates = dict()
    for line in file:
        image = list()
        elements = [e for e in line.split()]
        for E in range(1,len(elements)):
            e = int(elements[E])
            image.append(e)
        images.append(image)
    return images

# Counting the number of times each orientation occurs for a given set of images
# and returning that as counts dictionary 
def count_orintations(images):
    counts = dict()
    for image in images:
        orientation = image[0]
        if orientation not in counts:
            counts[orientation] = 0
        counts[orientation] += 1
    return counts

# Calculate the Gini impurity for a give set of images
def gini(images):
    impurity = 1
    counts = count_orintations(images)
    for orientation in counts:
        probabilty_of_label = counts[orientation]/len(images)
        impurity -=  probabilty_of_label**2
    return impurity

# Partition the given set images into true and false lists based on the threshold and
# the column/feature on which the partition is to be made
# and return both the lists
def partition(images, column, partition_value):
    true_values , false_values = list(), list()
    for image in images:
        if image[column] < partition_value:
            true_values.append(image)
        else:
            false_values.append(image)
    return true_values, false_values

# Find the best column/feature to split on based
# the best column is the one which gives the max information gain
# where info gain = original impurity - avg impurity of the two partitions
def find_best_split(images, columns, partition_value):
    max_gain = 0
    initial_impurity = gini(images)
    best_column = None
    for column in columns:
        T_values, F_values = partition(images, column, partition_value)

        weight = len(T_values)/(len(T_values) + len(F_values))
        avg_impurity = (weight * gini(T_values)) + ((1 - weight) * gini(F_values))
        info_gain = initial_impurity - avg_impurity
        if info_gain >= max_gain:
            max_gain = info_gain
            best_column = column
    return best_column, max_gain

# Build the tree till Depth = 6, recursively 
def build_tree(images, columns, partition_value, d):
    if d == 6:
        labels = count_orintations(images)
        predict = max(labels.items(), key=operator.itemgetter(1))[0]
        N = Node()
        N.prediction = predict
        return N
    best_column, max_gain = find_best_split(images, columns, partition_value)
    N = Node()
    N.column = best_column
    N.depth = d
    T_values, F_values = partition(images, best_column, partition_value)
    columns.remove(best_column)
    N.true = build_tree(T_values, columns, partition_value, d+1)
    N.false = build_tree(F_values, columns, partition_value, d+1)
    return N

# Test an input image based with the help of a given tree
# By treversing through the tree till leaf node and taking the prediction of leaf node
def test(test_image, root_node):
    global output
    if root_node.column == None:
        output.append(root_node.prediction)
    else:
        if test_image[root_node.column] < 127:
            test(test_image, root_node.true)
        else:
            test(test_image, root_node.false)

########## MAIN ###########

def train(path, model_file):
	
	my_images = read(path)
	My_decision_trees = list()

	for i in range(0,10):
		columns = [c for c in range(1,len(my_images[0]))]
		print("Building tree number %d" %(i+1))
		random_column_list = random.sample(columns,100)
		root = Node()
		root = build_tree(my_images, random_column_list, 127, 0)
		My_decision_trees.append(root)
		
	f = open(model_file, 'wb')
	pickle.dump(My_decision_trees, f, protocol=pickle.HIGHEST_PROTOCOL)
	f.close()
	
output = list()
def test_forest(input_file, model_file):
	global output
	test_images = read(input_file)
	f = open(model_file, 'rb')
	My_decision_trees = pickle.load(f)
	
	
	output_file = list()
	for tree in range(0,len(My_decision_trees)):
		
		output = list()
		for each in test_images: 
			test(each, My_decision_trees[tree])
		output_file.append(output)

	final_output = list()

	for i in range(0,len(output_file[0])):
		x = {0:0,90:0,180:0,270:0}
		for j in output_file:
			x[j[i]] += 1
		final_output.append(max(x.items(), key=operator.itemgetter(1))[0])

	count = 0
	
	for i in range(0,len(test_images)):
		if test_images[i][0] == final_output[i]:
			count += 1 
	Accuracy = 100*(count/len(test_images))
	print("Accuracy is %f" % (Accuracy))
	f.close()
	
	output = open("output.txt","w+")
	input = open(input_file)
	i = 0
	for line in input:
		image = list()
		elements = [e for e in line.split()]
		output.write(str(elements[0]) + ' ' + str(final_output[i])+'\n' )
		i += 1
	output.close()
	input.close()
