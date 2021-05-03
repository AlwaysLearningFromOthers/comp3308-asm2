import numpy as np
import os
import csv
from operator import itemgetter
from random import shuffle, seed

n_yes = 0
n_no = 0


def read_file():
    cwd = os.getcwd()
    with open(cwd + "/data/pima-indians-diabetes.csv") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    
    return data

#fold: how many folds in total
def train_test_split(data,fold):
    #seed(20210501)
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
def Euclidean_distance(train_data:list,test_data:list):
	sum = 0
	for i in range(0,len(test_data)-1):
		sum = sum + np.power(float(train_data[i])-float(test_data[i]),2)

	return np.sqrt(sum)


    
def KNN(k,train, test_data):
    select_distance = []
    for train_fold in train:
        for instance in train_fold:
            distance = Euclidean_distance(instance,test_data)
            select_distance.append({"distance":distance,"class":instance[-1]})

    selected  = []
    for dic in select_distance:
        #if there are not enough k instance in selected, add new ones in to it.
        if (len(selected) < k):
            selected.append(dic)
            selected = sorted(selected,key=itemgetter("distance"),reverse=True)
        # if the amount is more than k, compare 
        else:
            for new in selected:
                if dic["distance"] < new["distance"]:
                    new["distance"] = dic["distance"]
                    new["class"] = dic["class"]
                    break
            selected = sorted(selected,key=itemgetter("distance"),reverse=True)

    #detect label for test data
    n_yes = 0
    n_no = 0
    for one in selected:
        if one["class"] == "yes":
            n_yes += 1
        elif one["class"] == "no":
            n_no += 1
    if n_yes >= n_no:
        #return "yes"
        print("yes")
    else:
        #return "no"
        print("no")

if __name__ == "__main__":
    data = read_file()
    train_set, test_set = train_test_split(data,5)
    KNN(3,train_set,test_set[0])









