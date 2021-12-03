import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3 as sql

def Learn(weights, X_train, y_train, learning_rate, bias, epochs):
    # creating matrix of samples
    matrix = np.array(X_train)

    # starting learning
    for t in range(epochs):
        for i in range(len(X_train)):
            pred = 0
            value = np.dot(matrix[i], weights) + bias
            if value < 0:
                pred = 0
            else:
                pred = 1
            for j in range(len(weights)):
                weights[j] = weights[j] + learning_rate * (y_train[i] - pred) * matrix[i][j]
    return weights

def Test(weights, X_test, y_test, bias):
    # creating matrix of samples
    matrix = np.array(X_test)
    
    output = []
    # starting learning
    for i in range(len(X_test)):
        pred = 0
        value = np.dot(matrix[i], weights) + bias
        if value < 0:
            pred = 0
        else:
            pred = 1
        output.append(pred)

    print(f"Accuracy: {GetAccuracy(output, y_test)}%")

def GetAccuracy(output, y_test):
    num_correct=0
    for i in range(len(output)):
        if output[i]==y_test[i]:
            num_correct+=1
    accuracy = num_correct/len(output)
    return accuracy

if __name__ == "__main__":
    #creating dataframe
    data_df = pd.read_csv('high_diamond_ranked_10min.csv', header=0)

    #creating feature dataframe (X) and label (y)
    X = pd.DataFrame(data_df.iloc[:, 2:40])
    y = data_df['blueWins'].values

    #scaling features
    feature_scaler = StandardScaler()
    X = pd.DataFrame(feature_scaler.fit_transform(X))
    
    #split training set into cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    #the perceptron will have 38 input nodes
    #initializing random weights
    weights =[]
    
    #initializing the bias term
    bias = 0.4
    
    #initializing the learning rate to start
    learning_rate = 0.3
    
    #initilizing the number of iterations to loop for when learning
    epochs=100
    
    #saving the converged weights to file, so that every time the program
    #is run, the model is not trying to relearn the weights for the training
    #data
    try:
        with open("weights.npy","rb") as wf:
            print("Weights found, using old weights")
            npweights=np.load(wf)
            for weight in npweights:
                weights.append(weight)
            print(weights)
    except:
        with open("weights.npy","wb") as wf:
                print("No weights found, creating new")
                for i in range(38):
                    weights.append(np.random.randn() * 0.10)
                weights = Learn(weights, X_train, y_train, learning_rate, bias, epochs)
                np.save(wf, np.array(weights))

    #testing the model on the test set
    Test(weights, X_test,y_test, bias)
    
    #reporting the weights, and the top 3 most important weights (features)
    maxw = max(weights, key=abs)
    print(maxw)
    maxindex=weights.index(maxw)
    print(maxindex)
    print(data_df.keys()[maxindex+2])


