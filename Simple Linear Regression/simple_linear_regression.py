# Simple linear regression
import pandas
import numpy
import matplotlib.pyplot as plt
 
dataset = pandas.read_csv('Salary_Data.csv')

# Independent variable
x = dataset.iloc[:, :-1].values

# Dependent variable
y = dataset.iloc[:, 1].values

# Taking care of missing data

# Encoding categorical data
# Since y is dependent variable we don't need to state that their is
# no relation in between the encoded label values 0 and 1 for no and yes

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling since the linear regression class already does it, so we
# don't need to provide Feature scaling explicitly

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediciting the Test set results
y_predict = regressor.predict(x_test)

# Visualizing the Training set results i.e. using the same set on which the 
# machine learned
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results i.e. the dataset
# which we seperated into x_test and y_test
plt.scatter(x_test, y_test, color = 'red')
# Since our regressor model is already made and is based on a single unqiue
# model equation we can use either x_train or x_test to plot the same line
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show()
