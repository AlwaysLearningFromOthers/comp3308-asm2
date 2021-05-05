import math
import os
import sys
import csv
from operator import itemgetter

n_yes = 0
n_no = 0
def read_file(fname):
    cwd = os.getcwd()
    with open(cwd + "/" +fname) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    
    return data

#fold: how many folds in total
#create pima_folds.csv
def create_pima_folds_csv(data1,fold):
    #size = int(len(data))
    #fold_size = int(size / fold)
    fold = []
    temp = []

    no_data = []
    yes_data = []
    for data in data1:
        if data[len(data) - 1] == "no":
            no_data.append(data)
        else:
            yes_data.append(data)

    for i in range(0, 10):
        temp1 = []
        for j in range(0, 50):
            temp1.append(no_data[j + i * 50])
        if i == 8:
            for k in range(0, 26):
                temp1.append(yes_data[k + 216])
        elif i == 9:
            for k in range(0, 26):
                temp1.append(yes_data[k + 242])
        else:
            for k in range(0, 27):
                temp1.append(yes_data[k + i * 27])
        temp.append(temp1)

    with open('pima-folds.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(0, 10):
            writer.writerow(["fold" + str(i + 1)])
            for item in temp[i]:
                writer.writerow(item)
            writer.writerow("")

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
def Euclidean_distance(train_data, test_data):
    sum = 0
    for i in range(0, len(train_data) - 1):
        sum = sum + math.pow(float(train_data[i]) - float(test_data[i]), 2)
    return math.sqrt(sum)


# #KNN Model    
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
            sum_yes = sum_yes + math.pow(float(instance[col]) - mean[0], 2)
            num_yes += 1
        else:
            sum_no = sum_no + math.pow(float(instance[col]) - mean[1], 2)
            num_no += 1

    temp_yes = sum_yes / (num_yes - 1)
    temp_no = sum_no / (num_no - 1)
    sd_yes = math.sqrt(temp_yes)
    sd_no = math.sqrt(temp_no)
    sd.append(sd_yes)
    sd.append(sd_no)
    return sd

def probability_density(x, mean, sd):
    exp = -math.pow(float(x) - mean, 2) / (2 * math.pow(sd, 2))
    prob_dens = math.exp(exp) / (sd * math.sqrt(math.pi * 2))
    return prob_dens

#Naive Bayes Model
def NB(train,instance):
    #for every instance in test dataset
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
        print("no")
    else:
        print("yes")



# #performance of models
# def accuracy(train_set,test_set,model):
#     #split the data
#     label = [i[-1] for i in test_set]
#     num_same = 0
#     #prediction based on model
#     if str(model).endswith("NN"):  
#         k = int(model[0])
#         predict = KNN(k,train_set, test_set)
#     elif model == "NB":
#         predict = NB(train_set,test_set)
#     #accuracy og model
#     for i in range(0,len(predict)):
#             if predict[i] == label[i]:
#                 num_same += 1
#     perc = num_same/len(predict)
#     return perc

# def cross_validation(fname,model,fold):
#     percent_vector = []
#     for i in range(1,fold):
#         train_set,test_set = train_test_split(fname,i)
#         percentage = accuracy(train_set, test_set, model)
#         percent_vector.append(percentage)
#     return np.mean(percent_vector)


if __name__ == "__main__":
    train_set = read_file(sys.argv[1])
    test_set = read_file(sys.argv[2])
    model = str(sys.argv[3])
    if model.endswith("NN"):  
        k = int(model[0])
        for test_data in test_set:
            KNN(k,train_set, test_data)
    elif model.strip() == "NB":
        for test_data in test_set:
            NB(train_set,test_data)
    #create_pima_folds_csv(read_file("pima.csv"),10)