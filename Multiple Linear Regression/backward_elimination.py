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

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# for the constant b0 in the equation y = b0 + b1*x1 + ... + bn*xn
# we have b0*x0 where x0 = 1 thus we need to explicitly add it here for
# this case since the statsmodels library needs it for getting statistical
# information on our independent variables
 
# Thus we use np.ones(50,1).astype(int) to create an array of 50 rows using 
# numpy's  built in function ones and is astype to convert the array into an
# integer array

# Later since we need to add it to the beginning of the x dataset thus we append
# the x dataset to the newly created array of 50 rows and 1 column

# The axis = 1 denotes that we want to add the entities as columns for rows we use
# axis = 0

x = numpy.append(arr = numpy.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
# Summary gives detailed account of the p-value and other attributes, statistics
regressor_OLS.summary() # OLS = ordinary least squared ( i.e linear regression)

x_opt = x[:, [0, 1, 3, 4, 5]] # Removing x2 and refitting, since x2 has highest p-value
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 4, 5]] # Removing 1 and refitting, since 1 has highest p-value
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3, 5]] # Removing 4 and refitting, since 4 has highest p-value
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0, 3]] # Removing 5 and refitting, since 5 has highest p-value
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# whether to remove 5 or not since it is 0.06 which is very close to our SignificanceLevel(SL)
# which is 0.05 comes under improvement and optimization thus here we simply go on with using
# the Backward Elimination