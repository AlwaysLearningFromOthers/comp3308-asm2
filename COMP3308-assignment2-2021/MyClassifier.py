import numpy as np
import os
import sys
import csv
from operator import itemgetter
from random import shuffle, seed

n_yes = 0
n_no = 0
pima ="pima.csv"
pima_folds = "pima-folds"
def read_file(fname):
    cwd = os.getcwd()
    with open(cwd + "/" +fname) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    
    return data

#fold: how many folds in total
#create pima_folds.csv
def create_pima_folds_csv(data,fold):
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
    train = []
    test = dataset_split_data[0]
    for _data in dataset_split_data:
        train.append(_data)
    with open('pima-folds.csv', 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONE,escapechar = ' ')
        i = 1
        for fold in train:
            if i!=1:
                wr.writerow(["\n"+"fold"+str(i)])
                i+=1
                for row in fold:
                    wr.writerow(row)
            else:
                wr.writerow(["fold"+str(i)])
                i+=1
                for row in fold:
                    wr.writerow(row)

#train test dataset split
def train_test_split(fname,index):
    train = []
    test = []
    data = read_file(fname)
    for line in data:
        if ''.join(line).strip() != "":
            if ''.join(line).startswith('fold') and int(''.join(line)[-1])!= int(str(index)[-1]):
                set = 0
                continue
            if ''.join(line).startswith('fold') and int(''.join(line)[-1])== int(str(index)[-1]):
                set = 1
                continue
            if set==0:
                train.append(line)
            elif set ==1:
                test.append(line)
    return train, test




# train test are both vectors in Euclidean
def Euclidean_distance(train_data,test_data):
	sum = 0
	for i in range(0,len(train_data)-1):
		sum = sum + np.power(float(train_data[i])-float(test_data[i]),2)

	return np.sqrt(sum)


#KNN Model    
def KNN(k,train, test):
    predict = []
    for test_data in test:
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
            predict.append("yes")
        else:
            predict.append("no")
    return predict

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
    predict=[]
    for instance in test:
        m = []
        sd = []
        prob_yes = []
        prob_no = []
        for col in range(0,len(train[0])-1):
            m.append(mean(train, col))
            sd.append(standard_deviation(train,col,mean(train, col)))
        for col in range(0,len(train[0])-1):
            prob_yes.append(probability_density(instance[col],m[col][0],sd[col][0]))
            prob_no.append(probability_density(instance[col],m[col][1],sd[col][1]))
        #compare with train    
        num_yes = 0
        num_no = 0
        for instance in train:
            if instance[-1]=="yes":
                num_yes += 1
            else:
                num_no += 1
        #calculate probability for labels
        x_yes = num_yes/len(train)
        x_no = num_no/len(train)
        for p in prob_yes:
            if p != 0:
                x_yes = x_yes * p
        for p in prob_no:
            if p != 0:
                x_no = x_no * p
        #prediction labels
        if x_no > x_yes:
            predict.append("no")
        else:
            predict.append("yes")
    return predict


#performance of models
def accuracy(train_set:list,test_set:list,model:str):
    #split the data
    label = [i[-1] for i in test_set]
    num_same = 0
    #prediction based on model
    if str(model).endswith("NN"):  
        k = int(model[0])
        predict = KNN(k,train_set, test_set)
    elif model == "NB":
        predict = NB(train_set,test_set)
    #accuracy og model
    for i in range(0,len(predict)):
            if predict[i] == label[i]:
                num_same += 1
    perc = num_same/len(predict)
    return perc

def cross_validation(fname,model,fold):
    percent_vector = []
    for i in range(1,fold):
        train_set,test_set = train_test_split(fname,i)
        percentage = accuracy(train_set, test_set, model)
        percent_vector.append(percentage)
    return np.mean(percent_vector)

def print_prediction(predict):
    for pred in predict:
        print(pred)



if __name__ == "__main__":
    # acc = cross_validation("pima-folds.csv","1NN",10)
    # print(acc)
    train_set = read_file(sys.argv[1])
    test_set = read_file(sys.argv[2])
    model = str(sys.argv[3])
    if model.endswith("NN"):  
        k = int(model[0])
        predict = KNN(k,train_set, test_set)
        print_prediction(predict)
    elif model == "NB":
        predict = NB(train_set,test_set)
        print_prediction(predict)














