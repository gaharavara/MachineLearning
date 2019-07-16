# Multiple Linear Regession

# Impoting the libraries
import pandas
import numpy
import matplotlib.pyplot as plt

# Inserting the dataset
dataset = pandas.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Since we have city in the dataset thus we need to perform categorical encoding
# Use label encoder to encode the column into numbers
# To remove any relation between the numbers we get after encoding we use one
# hot encoder to create dummy variables

# Encoding categorical data
# Encoding the Independent variable ( situated in 3rd column)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3]) 
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap ( it is automatically done by the library, but 
# in case of some other places we need to do it explicitly) here removing
# the 0th column
x = x[:, 1:]

# Encoding must be done before splitting the dataset
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 0) 

# Feature Scaling ( Library takes care of it so we don't need to do it explicitly)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)