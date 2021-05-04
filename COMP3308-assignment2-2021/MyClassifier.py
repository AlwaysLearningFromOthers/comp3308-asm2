import numpy as np
import os
import sys
import csv
from operator import itemgetter
from random import shuffle, seed

n_yes = 0
n_no = 0

def read_file():
    cwd = os.getcwd()
    with open(cwd + "/pima.csv") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    
    return data

#fold: how many folds in total
#create pima_folds.csv
def create_pima_folds_csv(data:list,fold:int):
    seed(490581612)
    num = [i for i in range(len(data))]
    shuffle(num)
    step = len(data)//fold
    left = len(data)%fold
    dataset_split_index = [num[i:i+step] for i in range(0,len(num),step)]
    for i in range(0,left):
        dataset_split_index[i].append(dataset_split_index[-1][i])
    del dataset_split_index[-1]
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
    with open('pima-folds.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        i = 1
        for fold in train:
            wr.writerow(["fold"+str(i)])
            i+=1
            for row in fold:
                wr.writerow(row)

#train test dataset split
def train_test_split(fname,index):
    train = []
    test = []
    cwd = os.getcwd()
    with open(cwd+"/"+fname,"r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    for line in data:
        if ''.join(line).startswith('fold') and int(''.join(line)[-1])!= index:
            set = 0
            continue
        if ''.join(line).startswith('fold') and int(''.join(line)[-1])== index:
            set = 1
            continue
        if set==0:
            train.append(line)
        elif set ==1:
            test.append(line)
    return train, test




# train test are both vectors in Euclidean
def Euclidean_distance(train_data:list,test_data:list):
	sum = 0
	for i in range(0,len(test_data)-1):
		sum = sum + np.power(float(train_data[i])-float(test_data[i]),2)

	return np.sqrt(sum)


#KNN Model    
def KNN(k,train, test_data):
    select_distance = []
    
    for instance in train:
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
        print("yes")
    else:
        print("no")


# mean of yes and no for one column
def mean(data, col):
    sum_yes = 0
    sum_no = 0
    num_yes = 0
    num_no = 0
    mean = []
    for instance in data:
        if instance[-1] == "yes":
            sum_yes = sum_yes + float(instance[col])
            num_yes += 1
        else:
            sum_no = sum_no + float(instance[col])
            num_no += 1
    mean_yes = sum_yes / num_yes
    mean_no = sum_no / num_no
    mean.append(mean_yes)
    mean.append(mean_no)
    return mean

# sd of one column
def standard_deviation(data, col, mean):
    sum_yes = 0
    sum_no = 0
    num_yes = 0
    num_no = 0
    sd = []
    for instance in data:
        if instance[-1] == "yes":
            sum_yes = sum_yes + np.power(float(instance[col]) - mean[0], 2)
            num_yes += 1
        else:
            sum_no = sum_no + np.power(float(instance[col]) - mean[1], 2)
            num_no += 1

    temp_yes = sum_yes / (num_yes - 1)
    temp_no = sum_no / (num_no - 1)
    sd_yes = np.sqrt(temp_yes)
    sd_no = np.sqrt(temp_no)
    sd.append(sd_yes)
    sd.append(sd_no)
    return sd

def probability_density(x, mean, sd):
    exp = -np.power(float(x) - mean, 2) / (2 * np.power(sd, 2))
    prob_dens = np.exp(exp) / (sd * np.sqrt(np.pi * 2))
    return prob_dens

#Naive Bayes Model
def NB(train,test):
    #for every instance in test dataset
    for instance in test:
        m = []
        sd = []
        prob_yes = []
        prob_no = []

        for col in range(0,len(instance)):
            m.append(mean(train, col))
            sd.append(standard_deviation(train,col,mean(train, col)))

        for col in range(0,len(instance)):
            prob_yes.append(probability_density(instance[col],m[col][0],sd[col][0]))
            prob_no.append(probability_density(instance[col],m[col][1],sd[col][1]))

        num_yes = 0
        num_no = 0
        for instance in train:
            if instance[-1]=="yes":
                num_yes += 1
            else:
                num_no += 1

        x_yes = num_yes/len(train)
        x_no = num_no/len(train)
        for p in prob_yes:
            if p != 0:
                x_yes = x_yes * p
        for p in prob_no:
            if p != 0:
                x_no = x_no * p

        if x_no > x_yes:
            print("no")
        else:
            print("yes")




if __name__ == "__main__":
    data = read_file()
    train_set, test_set = train_test_split("pima-folds.csv",1)











