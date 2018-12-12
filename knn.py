#!/usr/bin/env python
import pickle
import sys
import numpy as np
import copy


'''  
The k-nearest neighbor algorithm, as the name suggests predicts the class based on the 'k' nearest neighbors by comparing the distances (in our case, Euclidean distance)


In our problem, we first calculate the euclidean distance of each of the test data with each of the train data.
Then, we take first 'k' values of the sorted list of euclidean distances. We consider the orientations of these k euclidean distances for the final orientation prediction. 





'''

#train_all = open("train-data.txt", "r")
#test_all = open("test-data.txt", "r")


#The function below reads the input data and returns that data as an array.
def read_file(file):
    data = []
    data_list = []
    for line in file:
        data.append([int(dot) for dot in line.split()[1:]])    #Takes the data except for the label and appends in the list
        data_list.append(line.split()[0:2])                    #Appends the orientation
    return np.array(data),np.array(data_list)

#train_data,train_names=read_file(train_all)
#test_data,test_names=read_file(test_all)	


#The eucidean function calculates the euclidean distance between the data points 
def euclidean(train_data,test_data):
	count = 0
	#The distance of the test data with each of the training data is calculated and stores in a list.
	euc_distances = []
	for test in test_data:
		distances = []
		count += 1
		#print(count)
		for train in train_data:
			dist = np.linalg.norm(test[1:]-train[1:])
			distances.append(dist)
		euc_distances.append(distances)
	return euc_distances

# The following function 'knn' was referred from 'Sanket Patole' 
#knn predicts the orientation by taking the first 'k' values from the sorted euclidean distances.


def knn(euc_distances, train_data):	
	k = 11
	final_minimum = []
	final_index = []
	new_distances = copy.deepcopy(euc_distances)     #So that, it does not overwrite the original distances.
	for list1 in new_distances:
	    minimum = []
	    min_index = []
	    copied = copy.copy(list1)
	    for i in range(0, k):                     # Gives first 'k' minimum euclidean distances.
	        min1 = 9999999
	        for j in range(len(copied)):  
	            if copied[j] < min1:
	                min1 = copied[j]        
	        minimum.append(min1)
	        ind = list1.index(min1)
	        min_index.append(ind)
	        copied.remove(min1)
	    final_minimum.append(minimum)
	    final_index.append(min_index)
	final_orientation = []
	orientation = []

	for k in final_index:
		orientation = []
		for i in k:
			orientation.append(train_data[i][0])
		final_orientation.append(orientation)
	new_orientation = []
	for i in final_orientation:                  #Returns the orientations based on the indices of the minimum euclidean distances of the data.
		results = list(map(int, i))
		new_orientation.append(results)
	final_ori = []
	for i in new_orientation:
		x = max(i,key=i.count)                    #Returns the orientation based on the list of maximum number of counts of the orientations.
		final_ori.append(x)
	prediction = [str(x) for x in final_ori]
	return prediction

# accuracy calculates the accuracy of the orientations predicted correctly to the total number of the data.

def accuracy(prediction, test_names):
    f = open('output.txt', 'a')
    accuracy = 0.0
    output1 = []
    for i in range(0,len(test_names)):
            temp_op = []
            a = prediction[i]
            b = test_names[i][0]
            temp_op.append(b)
            temp_op.append(a)
            output1.append(temp_op)
            f.write(test_names[i][0]+' '+prediction[i]+'\n')
            if test_names[i][1] == prediction[i]:
                    accuracy += 1
    accuracy = (float(accuracy)/len(test_names)) * 100 
    print("Accuracy = ",accuracy,"%")
    f.close()
"""
	with open('answer.txt', 'w') as f:
		for item in output1:
			f.write("%s\n" % item)
	with open('output.txt', 'w') as f:
		for _list in output1:
			for _string in _list:
				f.write(str(_string))
"""
#accuracy(knn(euclidean(train_data,test_data), train_data), test_names)


