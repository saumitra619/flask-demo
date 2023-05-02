# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))
