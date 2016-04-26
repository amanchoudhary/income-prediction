import csv
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from feature_similarity import feature_correlation
from config import METHOD
import numpy as np
from neural_network import make_neural_network
from missing_values import fill_missing_values

TESTING_SET_LIMIT = 15000

def convert_to_numpy(train_vector):
    result = []
    for row in train_vector:
        list = np.array(row, dtype=np.float32)
        result.append(list)
    return np.array(result, dtype=np.float32)

def get_label_value(label):
    label = np.array(label, dtype=np.int32)
    return (np.arange(2) == label[:,None]).astype(np.float32)


def get_inputs(filename):
    file = open(filename, 'rb')
    input = list(csv.reader(file))
    return input

def divide_dataset(features, labels):
    i = 0
    test_features = []
    test_labels = []
    while i <= TESTING_SET_LIMIT:
        index = random.randint(0, len(features)-1)
        test_features.append(features[index])
        test_labels.append(labels[index])
        del features[index]
        del labels[index]
        i += 1
    return features, labels, test_features, test_labels

def process(input, dict, dict_start, index):
    if (dict[index].has_key(input)):
        return dict[index][input]
    dict[index][input] = dict_start[index]
    dict_start[index] += 1.0
    return dict[index][input]

def normalize_feature(result, index):
    min_value = 100000000.0
    max_value = -100000000.0
    for row in result:
        val = float(row[index])
        if val < min_value:
            min_value = val
        if val > max_value:
            max_value = val

    i = 0
    while i < len(result):
        result[i][index] = (float(result[i][index])-min_value)/(max_value-min_value)
        i += 1

def process_input_files(inputs):
    result = []
    for i in range(0, len(inputs)):
        feature_example = [1]*11
        result.append(feature_example)

    dictionary_list = []
    dictionary_start = [0.0]*14
    for i in range(0, 14):
        dict = {}
        dictionary_list.append(dict)

    labels = [0]*len(inputs)
    i = 0

    label_dict = {}
    label_list = [label_dict]
    label_start = [0.0]*1

    while i < len(inputs):
        # Age feature
        result[i][0] = float(inputs[i][0])
        # Workclass feature
        result[i][1] = process(inputs[i][1], dictionary_list, dictionary_start, 1)
        # Education feature
        result[i][2] = process(inputs[i][3], dictionary_list, dictionary_start, 3)
        # Country Feature
        result[i][3] = process(inputs[i][13], dictionary_list, dictionary_start, 13)
        # Martial Status feature
        result[i][4] = process(inputs[i][5], dictionary_list, dictionary_start, 5)
        # Occupation feature
        result[i][5] = process(inputs[i][6], dictionary_list, dictionary_start, 6)
        # Relationship feature
        result[i][6] = process(inputs[i][7], dictionary_list, dictionary_start, 7)
        # Race feature
        result[i][7] = process(inputs[i][8], dictionary_list, dictionary_start, 8)
        # Gender feature
        result[i][8] = process(inputs[i][9], dictionary_list, dictionary_start, 9)
        # Capital feature
        result[i][9] = float(inputs[i][10])-float(inputs[i][11])
        # Hours per week feature
        result[i][10] = float(inputs[i][12])


        labels[i] = process(inputs[i][14], label_list, label_start, 0)
        i += 1
    print label_list[0]
    i = 0
    while i < len(result[0]):
        normalize_feature(result, i)
        i += 1
    return result, labels


def clean_data(input):
    dict = {}

    dict[' State-gov'] = 'gov'
    dict[' Local-gov'] = 'gov'
    dict[' Without-pay'] = ' Never-worked'
    dict[' 7th-8th'] = 'dropout'
    dict[' 1st-4th'] = 'dropout'
    dict[' 9th'] = 'dropout'
    dict[' 5th-6th'] = 'dropout'
    dict[' 10th'] = 'dropout'
    dict[' 11th'] = 'dropout'
    dict[' 12th'] = 'dropout'
    dict[' Preschool'] = 'dropout'
    dict['HS-grad'] = ' dropout'
    dict[' Assoc-voc'] = 'assoc'
    dict[' Assoc-acdm'] = 'assoc'
    dict[' Masters'] = ' Bachelors'
    dict[' Separated'] = 'divored'
    dict[' Divorced'] = 'divored'
    dict[' Married-spouse-absent'] = 'divored'
    dict[' Married-AF-spouse'] = 'married'
    dict[' Married-civ-spouse'] = 'married'
    dict[' Armed-Forces'] = 'red'
    dict[' Craft-repair'] = 'green'
    dict[' Other-service'] = 'green'
    dict[' Transport-moving'] = 'green'
    dict[' Prof-specialty'] = 'blue'
    dict[' Sales'] = 'blue'
    dict[' Machine-op-inspct'] = 'green'
    dict[' Exec-managerial'] = 'blue'
    dict[' Protective-serv'] = 'blue'
    dict[' Handlers-cleaners'] = 'green'
    dict[' Adm-clerical'] = 'green'
    dict[' Tech-support'] = 'blue'
    dict[' Protective-serv'] = 'blue'
    dict[' Farming-fishing'] = 'green'
    dict[' Priv-house-serv'] = 'green'
    dict[' Ireland'] = 'europe'
    dict[' Peru'] = 'africa'
    dict[' Loas'] = 'south-east-asia'
    dict[' Ecuador'] = 'south-america'
    dict[' Cambodia'] = 'south-east-asia'
    dict[' France'] = 'europe'
    dict[' Scotland'] = 'europe'
    dict[' Italy'] = 'europe'
    dict[' Hong'] = 'asia'
    dict[' Canada'] = 'north-america'
    dict[' Nicaragua'] = 'south-america'
    dict[' Japan'] = 'asia'
    dict[' Taiwan'] = 'asia'
    dict[' Greece'] = 'europe'
    dict[' India'] = 'asia'
    dict[' Yugoslavia'] = 'europe'
    dict[' Jamaica'] = 'south-america'
    dict[' England'] = 'europe'
    dict[' Portugal'] = 'europe'
    dict[' Mexico'] = 'south-america'
    dict[' Trinadad&Tobago'] = 'south-america'
    dict[' Honduras'] = 'south-america'
    dict[' South'] = 'europe'
    dict[' Vietnam'] = 'south-east-asia'
    dict[' Poland'] = 'europe'
    dict[' Philippines'] = 'europe'
    dict[' Hungary'] = 'europe'
    dict[' Columbia'] = 'south-america'
    dict[' Germany'] = 'europe'
    dict[' Thailand'] = 'south-east-asia'
    dict[' Haiti'] = 'south-america'
    dict[' Dominican-Republic'] = 'south-america'
    dict[' China'] = 'asia'
    dict[' United-States'] = 'north-america'
    dict[' Holand-Netherlands'] = 'europe'
    dict[' Guatemala'] = 'south-america'
    dict[' El-Salvador'] = 'south-america'
    dict[' Puerto-Rice'] = 'south-america'
    dict[' Cuba'] = 'south-america'
    dict[' Iran'] = 'asia'


    for i in range(0, len(input)):
        for j in range(0, len(input[i])):
            if dict.has_key(input[i][j]):
                input[i][j] = dict[input[i][j]]
    return input


def print_distinct(input, index):
    dict = {}
    for row in input:
        if dict.has_key(row[index]):
            dict[row[index]] += 1
        else:
            dict[row[index]] = 1
    print dict

train_input = get_inputs('train.csv')
test_input = get_inputs('test.csv')
input = train_input + test_input

input = clean_data(input)
# print_distinct(input, 1)
input = fill_missing_values(input)

feature, label = process_input_files(input)

for i in range(0, len(label)):
    label[i] = int(label[i])

train_feature, train_label, test_feature, test_label = divide_dataset(feature, label)

if METHOD is 'GradientBoostingClassifier':
    model = GradientBoostingClassifier()
    model.fit(train_feature, train_label)
    result = model.predict(test_feature)
    sum = 0
    for i in range(0, len(result)):
        if result[i] == test_label[i]:
            sum += 1

    print float(sum*100)/float(len(result))
else:
    train_feature = convert_to_numpy(train_feature)
    test_feature = convert_to_numpy(test_feature)
    train_label = get_label_value(train_label)
    test_label = get_label_value(test_label)
    make_neural_network(train_feature, train_label, test_feature, test_label, 2, 11)
