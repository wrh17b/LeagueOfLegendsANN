import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

"""
Psudo code for algo:
Let X be the set of training data
Init weight with random values
repeat
    for each training example X[i] do
        compute predicted output yk[i]
        for each weight w[j] do
            update weight w[j] = w[j] + learning rate * prediction error * val of jth attr of training example xi
        end weight update for
    end training example for
uuntil stopping condition met 
notes: 
learning weight is val btwn 0 and 1 used to influence val of old weight
prediction error is 
"""
print(len(weights))

#initializing the bias term
bias = 0.4

#initializing the learning rate to start
learning_rate = 0.5

#creating matrix of samples
matrix = np.array(X)
print(matrix)

output = []

#starting learning
for i in range(len(X)):
    pred = 0
    error = None
    value = np.dot(matrix[i], weights) + bias
    if value < 0:
        pred = 0
    else:
        pred = 1
    for j in range (len(weights)):
        weights[j]=weights[j] + learning_rate*(y[i]-pred)*matrix[i][j]

# Learning should be complete
#we should do some validation
