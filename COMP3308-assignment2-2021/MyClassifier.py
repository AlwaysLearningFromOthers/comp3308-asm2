import numpy as np
import os
import csv
from random import shuffle, seed



def read_file():
    cwd = os.getcwd()
    with open(cwd + "/data/pima-indians-diabetes.csv") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    
    return data

#fold: how many folds in total
def train_test_split(data,fold):
    seed(20210501)
    num = [i for i in range(768)]
    shuffle(num)
    dataset_split_index = [num[i:i+fold] for i in range(0,len(num),fold)]
    
    dataset_split_data = []
    set_data = []
    for set_num in dataset_split_index:
        for j in set_num:
            set_data.append(data[j])
        dataset_split_data.append(set_data)
        set_data = []
    test = []
    train = []
    test = dataset_split_data[0]
    for _data in dataset_split_data:
        train.append(_data)
    return train, test
    



# train test are both vectors in Euclidean
def Euclidean_distance(train_data,test_data):
	sum = 0
	for i in range(0,len(test_data)):
		sum = sum + np.power(train_data[i]-test_data[i],2)

	return np.sqrt(sum)

if __name__ == "__main__":
    data = read_file()
    







