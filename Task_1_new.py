import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree, neural_network
from sklearn import linear_model
from sklearn import svm

feature_1 = np.zeros((10, 6))
feature_2 = np.zeros((10, 6))
feature_3 = np.zeros((10, 6))
feature_4 = np.zeros((10, 6))



def Model_X():
    #clf = linear_model.LinearRegression()
    clf =  neural_network.MLPRegressor(tol=.01, hidden_layer_sizes = 4)
    return clf

def Model_Y():
    #clf = linear_model.LinearRegression()
    clf = neural_network.MLPRegressor(tol=.01, hidden_layer_sizes = 4)
    return clf



def classify(data):
    spikes = 0
    isspikes = True
    for row in data:
        if row[7] == "5" and row[8] == "15":
              if row[9] == 1:
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

        if row[7] == "10.5" and row[8] == "15":
              if row[9] == 1:
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

        if row[7] == "5" and row[8] == "4.5":
              if row[9] == 1:
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

        if row[7] == "10.5" and row[8] == "4.5":
              if row[9] == 1:
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
        element = row.split("\t")
        dataset.append(element)
    labelDataset =[]
    for data in dataset:
        if int(data[0]) > thresold:
            data.append(1)
        else:
            data.append(0)
        labelDataset.append(data)


    return labelDataset

def meansquareerror(observed_x, observed_y, predicted_x, predicted_y):
    sum = 0
    for o_x, o_y, p_x, p_y  in zip(observed_x, observed_y, predicted_x, predicted_y ):
        sum = sum + np.sqrt((o_x-p_x)**2 + (o_y-p_y)**2)

    return sum / len(observed_x)

def main():
    datasets = []
    File = open("dataset_new.txt", "r")
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
    points_x = [5,5,5,5,5,5,5,5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,5,5,5,5,5,5,5,5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5]
    points_y = [15,15,15,15,15,15,15,15,5,5,5,5,5,5,5,5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5]



    model_x = Model_X()
    model_y = Model_Y()

    model_x.fit(trainingData, points_x)
    model_y.fit(trainingData, points_y)

    predicted_x = model_x.predict(testingData)
    predicted_y = model_y.predict(testingData)
    point = []
    for x, y in zip (predicted_x, predicted_y):
        point.append((x,y))
    print (point)

    observed_x= {5,5,10.5,10.5,5,5,10.5,10.5}
    observed_y = {15, 15, 5, 5, 4.5, 4.5, 4.5, 4.5}
    print("MSE = ", meansquareerror(observed_x, observed_y, predicted_x, predicted_y))


main()