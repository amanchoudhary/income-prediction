from sklearn.ensemble import GradientBoostingClassifier

def process(input, dict, dict_start, index):
    if (dict[index].has_key(input)):
        return dict[index][input]
    dict[index][input] = dict_start[index]
    dict_start[index] += 1.0
    return dict[index][input]

def get_value(val, dict):
    for key,value in dict.iteritems():
        if value == val:
            return key

def normalize_feature(feature_1, feature_2):
    feature = feature_1 + feature_2

    dictionary_list = []
    dictionary_start = [0.0]*3
    for i in range(0, 3):
        dict = {}
        dictionary_list.append(dict)

    for i in range(0, len(feature)):
        feature[i][0] = float(feature[i][0])
        feature[i][4] = float(feature[i][4])
        feature[i][5] = float(feature[i][5])

        feature[i][1] = process(feature[i][1], dictionary_list, dictionary_start, 0)
        feature[i][2] = process(feature[i][2], dictionary_list, dictionary_start, 1)
        feature[i][3] = process(feature[i][3], dictionary_list, dictionary_start, 2)


    feature_1 = feature[:len(feature_1)]
    feature_2 = feature[-len(feature_2):]
    return feature_1, feature_2

def fill_missing_values(data):
    train_feature_workclass = []
    train_label_workclass = []

    train_feature_country = []
    train_label_country = []

    test_feature_workclass = []
    test_feature_country = []


    for i in range(0, len(data)):
        temp = []
        temp.append(data[i][0])
        temp.append(data[i][3])
        temp.append(data[i][8])
        temp.append(data[i][9])
        temp.append(data[i][10])
        temp.append(data[i][12])

        if data[i][1] == ' ?':
            test_feature_workclass.append(temp)
        else:
            train_feature_workclass.append(temp)
            train_label_workclass.append(data[i][1])


        if data[i][13] == ' ?':
            test_feature_country.append(temp)
        else:
            train_feature_country.append(temp)
            train_label_country.append(data[i][13])

    train_feature_country, test_feature_country = normalize_feature(train_feature_country, test_feature_country)
    train_feature_workclass, test_feature_workclass = normalize_feature(train_feature_workclass, test_feature_workclass)

    dict_country = {}
    dict_workclass = {}

    country_start = 0
    workclass_start = 0

    for i in range(0, len(train_label_country)):
        if dict_country.has_key(train_label_country[i]):
            train_label_country[i] = dict_country[train_label_country[i]]
        else:
            dict_country[train_label_country[i]] = country_start
            train_label_country[i] = dict_country[train_label_country[i]]
            country_start += 1

    for i in range(0, len(train_label_workclass)):
        if dict_workclass.has_key(train_label_workclass[i]):
            train_label_workclass[i] = dict_workclass[train_label_workclass[i]]
        else:
            dict_workclass[train_label_workclass[i]] = workclass_start
            train_label_workclass[i] = dict_workclass[train_label_workclass[i]]
            workclass_start += 1



    model_workplace = GradientBoostingClassifier()
    model_workplace.fit(train_feature_workclass, train_label_workclass)


    model_country = GradientBoostingClassifier()
    model_country.fit(train_feature_country, train_label_country)

    result_workplace = model_workplace.predict(test_feature_workclass)
    result_country = model_country.predict(test_feature_country)

    i = 0
    j = 0


    for index in range(0, len(data)):
        if data[index][1] == ' ?':
            data[index][1] = get_value(result_workplace[i], dict_workclass)
            i += 1
        if data[index][13] == ' ?':
                data[index][13] = get_value(result_country[j], dict_country)
                j += 1


    return data


