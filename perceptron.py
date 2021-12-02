import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#creating dataframe
data_df = pd.read_csv('high_diamond_ranked_10min.csv', header=0)

#creating feature dataframe (X) and label (y)
X = pd.DataFrame(data_df.iloc[:, 2:40])
y = data_df['blueWins'].values

#scaling features
feature_scaler = StandardScaler()
X = pd.DataFrame(feature_scaler.fit_transform(X))

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

#creating matrix of samples
matrix = np.array(X)
print(matrix)

output = []
#starting learning
for i in range(38):
    pred = 0
    error = None
    value = np.dot(matrix[i], weights) + bias
    if value < 0:
        pred = 0
    else:
        pred = 1
