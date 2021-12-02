import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def Learn(weights, X_train, y_train, learning_rate):
    # creating matrix of samples
    matrix = np.array(X_train)
    print(matrix)

    output = []

    # starting learning
    for i in range(len(X_train)):
        pred = 0
        error = None
        value = np.dot(matrix[i], weights) + bias
        if value < 0:
            pred = 0
        else:
            pred = 1
        output.append(pred)
        for j in range(len(weights)):
            weights[j] = weights[j] + learning_rate * (y_train[i] - pred) * matrix[i][j]
    return weights

def Test(weights, X_test, y_test):

    # creating matrix of samples
    matrix = np.array(X_test)
    print(matrix)

    output = []

    # starting learning
    for i in range(len(X_test)):
        pred = 0
        error = None
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #Weight vector


    print(X)

    #the perceptron will have 38 input nodes

    #initializing random weights
    weights = []
    for i in range(38):
        weights.append(np.random.randn() * 0.10)

    print(weights)

    print(len(weights))

    #initializing the bias term
    bias = 0.4

    #initializing the learning rate to start
    learning_rate = 0.3


    # Learning should be complete

    weights = Learn(weights, X_train, y_train, learning_rate)

    #we should do some validation

    Test(weights, X_test,y_test)
