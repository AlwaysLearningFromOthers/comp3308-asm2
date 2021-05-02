import numpy as np
import os
import csv
from random import shuffle, seed



def read_file():
    cwd = os.getcwd()
    with open(cwd + "/data/pima-indians-diabetes.csv") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    
    return rows

#step: how many folds in total
def train_test_split(step):
    seed(20210501)
    num = [i for i in range(768)]
    shuffle(num)
    b = [num[i:i+step] for i in range(0,len(num),step)]




# train test are both vectors in Euclidean
def Euclidean_distance(train_data,test_data):
	sum = 0
	for i in range(0,len(test_data)):
		sum = sum + np.power(train_data[i]-test_data[i],2)

	return np.sqrt(sum)

if __name__ == "__main__":
    read_file()







