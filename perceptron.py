import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

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
    # starting testing
    for i in range(len(X_test)):
        pred = 0
        value = np.dot(matrix[i], weights) + bias
        if value < 0:
            pred = 0
        else:
            pred = 1
        output.append(pred)

    print(f"Model Accuracy: {GetAccuracy(output, y_test) * 100}%\n")


def GetAccuracy(output, y_test):
    num_correct = 0
    for i in range(len(output)):
        if output[i] == y_test[i]:
            num_correct += 1
    accuracy = num_correct / len(output)
    return accuracy


if __name__ == "__main__":
    # creating dataframe
    data_df = pd.read_csv('high_diamond_ranked_10min.csv', header=0)

    # creating feature dataframe (X) and label (y)
    X = pd.DataFrame(data_df.iloc[:, 2:40])
    y = data_df['blueWins'].values

    # scaling features
    feature_scaler = StandardScaler()
    X = pd.DataFrame(feature_scaler.fit_transform(X))

    # split training set into cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # the perceptron will have 38 input nodes
    weights = []

    # initializing the bias term
    bias = 0.9

    # initializing the learning rate to start
    learning_rate = 0.7

    # initializing the number of iterations to loop for when learning
    epochs = 100

    # saving the converged weights to file, so that every time the program
    # is run, the model is not trying to relearn the weights for the training
    # data
    try:
        with open("weights.npy", "rb") as wf:
            npweights = np.load(wf)
            for weight in npweights:
                weights.append(weight)
    except:
        with open("weights.npy", "wb") as wf:
            for i in range(38):
                weights.append(np.random.randn() * 0.10)
            weights = Learn(weights, X_train, y_train, learning_rate, bias, epochs)
            np.save(wf, np.array(weights))

    # testing the model on the test set
    Test(weights, X_test, y_test, bias)

    # reporting the weights, and the top 3 most important weights (features)
    print("Learned Weights:")
    for i in range(len(weights)):
        print(f"{i}, {weights[i]}")
    print()

    print("Top Three Most Important Features:")
    sortWeights = []
    for i in range(len(weights)):
        sortWeights.append(weights[i])
    sortWeights.sort(reverse=True)
    first = weights.index(sortWeights[0])
    second = weights.index(sortWeights[1])
    third = weights.index(sortWeights[2])

    print(data_df.keys()[first + 2])
    print(data_df.keys()[second + 2])
    print(data_df.keys()[third + 2])

    #printing the value of sklearn's perceptron to compare with our model
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_test, y_test)
    score = clf.score(X_test, y_test)
    print(f"\nSklearn Perceptron's Accuracy: {score * 100}%")


