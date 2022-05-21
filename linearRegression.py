# linear regression is a machine learning algorithm that deals with prediction of integer values
# It impliment the equation y = mx + c of an equation.
# in this part i am going to use it to try and predict the chance of a student passing his/her exams given certain parameters
# I am using students data from UCI institute.

# Load all the libraries
from matplotlib import style
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

# Loading the data
data = pd.read_csv('student-mat.csv', sep=';')

dt = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


X = np.array(dt.drop(["G3"], 1))
y = np.array(dt["G3"])
# plotting data using scatterplot

# style.use("ggplot")
# plt.scatter(X, y)
# plt.xlabel("Features data")
# plt.ylabel("Output")
# plt.show()

# spliting data into test and training datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LinearRegression()
# model.fit(x_train, y_train)
# acc = model.score(x_test, y_test)

# # saving model
# with open('student.pickle', "wb") as f:
#     pickle.dump(model, f)

pickle_in = open("student.pickle", "rb")
model = pickle.load(pickle_in)
prediction = model.predict(x_test)
for x in range(len(prediction)):
    print("predicted: ", prediction[x], "data: ", x_test[x], "Actual: ", y_test[x])