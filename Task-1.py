import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import linear_model

feature_1 = np.zeros((10, 6))
feature_2 = np.zeros((10, 6))
feature_3 = np.zeros((10, 6))
feature_4 = np.zeros((10, 6))



def Model():
    clf = tree.DecisionTreeClassifier(max_depth=100)
    return clf


def classify(data):
    spikes = 0
    isspikes = True
    for row in data:
        if row[7] == "1":
              if row[8] == 1:
                  if isspikes:
                      spikes = spikes + 1
                      isspikes = False
                  if(int(row[0]) > feature_1[spikes-1][0]):
                      for index in range(6):
                          feature_1[spikes-1][index] = row[index]
              else:
                  isspikes =  True
                  if spikes == 10:
                      spikes = 0

        if row[7] == "2":
              if row[8] == 1:
                  if isspikes:
                      spikes = spikes + 1
                      isspikes = False
                  if(int(row[0]) > feature_2[spikes-1][0]):
                      for index in range(6):
                          feature_2[spikes-1][index] = row[index]
              else:
                  isspikes =  True
                  if spikes == 10:
                      spikes = 0

        if row[7] == "3":
              if row[8] == 1:
                  if isspikes:
                      spikes = spikes + 1
                      isspikes = False
                  if(int(row[0]) > feature_3[spikes-1][0]):
                      for index in range(6):
                          feature_3[spikes-1][index] = row[index]
              else:
                  isspikes =  True
                  if spikes == 10:
                      spikes = 0

        if row[7] == "4":

              if row[8] == 1:
                  if isspikes:
                      spikes = spikes + 1
                      isspikes = False
                  if(int(row[0]) > feature_4[spikes-1][0]):
                      for index in range(6):
                          feature_4[spikes-1][index] = row[index]
              else:
                  isspikes =  True
                  if spikes == 10:
                      spikes = 0
    return feature_1, feature_2, feature_3, feature_4

def train(datasets):
    thresold = 200
    dataset = []
    for row in datasets:
        row = row.rstrip('\n')
        element = row.split(",")
        dataset.append(element)
    labelDataset =[]
    for data in dataset:
        if int(data[0]) > thresold:
            data.append(1)
        else:
            data.append(0)
        labelDataset.append(data)


    return labelDataset

def meansquareerror(observed, predicted):
    sum = 0
    for index in range(len(observed)):
        sum = sum + (observed[index] - predicted[index])**2

    return sum / len(observed)

def main():
    datasets = []
    File = open("dataset.txt", "r")
    for line in File:
        datasets.append(line)


    trainedData = train(datasets)
    feature_1, feature_2, feature_3, feature_4 = classify(trainedData)

    feature_1_trainging, feature_1_testing = train_test_split(feature_1, test_size=0.2)
    feature_2_trainging, feature_2_testing = train_test_split(feature_2, test_size=0.2)
    feature_3_trainging, feature_3_testing = train_test_split(feature_3, test_size=0.2)
    feature_4_trainging, feature_4_testing = train_test_split(feature_4, test_size=0.2)

    trainingData = np.vstack((feature_1_trainging, feature_2_trainging, feature_3_trainging, feature_4_trainging))
    testingData = np.vstack((feature_1_testing, feature_2_testing, feature_3_testing, feature_4_testing))
    points = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3]

    model = Model()
    model.fit(trainingData, points)
    y = [[np.sqrt(5*5 + 15 * 15)], [np.sqrt(10.5*10.5 + 15*15)], [np.sqrt(5*5 + 4.5*4.5)], [np.sqrt(10.5*10.5 + 4.5*4.5)]]
    X = [[0,0],[1,1], [2,2], [3,3]]
    reg = linear_model.BayesianRidge()
    reg.fit(X, y)
    data = model.predict(testingData)
    feature_data = np.zeros((8,2), dtype=int)
    for index, point in enumerate(data):
        feature_data[index] = [data[index], data[index]]

    print(reg.predict(feature_data))

    observed = [np.sqrt(5*5 + 15 * 15),np.sqrt(5*5 + 15 * 15), np.sqrt(10.5*10.5 + 15*15),np.sqrt(10.5*10.5 + 15*15),
                np.sqrt(5*5 + 4.5*4.5),np.sqrt(5*5 + 4.5*4.5), np.sqrt(10.5*10.5 + 4.5*4.5),np.sqrt(10.5*10.5 + 4.5*4.5)]

    predicted = reg.predict(feature_data)

    print("MSE = ", meansquareerror(observed, predicted))


main()