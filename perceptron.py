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
