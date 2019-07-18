# Polynomial Regression
import pandas
import numpy
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pandas.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No splitting since we want to predict what the salary will be
# for the said eployee level, thus we will only build our polynomial
# regression model

# No need to apply feature scaling since the library we are using will
# do the job

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Here we modify the x dataset so that we get x^2 value for x (i.e.
# polynomial form ), also it includes adding x0 also to the dataset
# sincw degree = 2 is the default value thus we keep degree as 2
poly_reg = PolynomialFeatures(degree = 2)
# We fit our newly created poly_reg with x and have their value inside
# x_poly
x_poly = poly_reg.fit_transform(x)

# Now making model for the polynomial dataset x_poly which we got from the
# previous 3 lines
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
