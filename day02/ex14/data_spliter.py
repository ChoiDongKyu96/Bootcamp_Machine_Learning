# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_spliter.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/28 15:08:21 by dochoi            #+#    #+#              #
#    Updated: 2020/05/28 16:33:20 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the traning set., →
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1. proportion: has to be a float,
    the proportion of the dataset that will be assigned to the training set., →
    Returns: (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray None if x or y is an empty numpy.ndarray.
    None if x and y do not share compatible dimensions.
    Raises: This function should not raise any Exception. """
    n = int(float(x.shape[0] * proportion))
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    x_train = x[0:n]
    x_test = x[n:]
    y_train = y[0:n]
    y_test = y[n:]
    return (x_train, x_test, y_train, y_test)

def zscore(x):
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = math.sqrt(var/ (len(x) - 1))
    return (x - mu) / std

csv_data = pd.read_csv("../resources/are_blue_pills_magics.csv")
y_n = []
x = np.array(csv_data["Micrograms"]).reshape(-1,1)
x = zscore(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = zscore(y)

temp = data_spliter(x, y, 0.5)
x_train = temp[0]
x_test = temp[1]
y_train = temp[2]
y_test = temp[3]
print(temp)
x2 = add_polynomial_features(x_train, 2)
x3 = add_polynomial_features(x_train, 3)
x4 = add_polynomial_features(x_train, 4)
x5 = add_polynomial_features(x_train, 5)
x6 = add_polynomial_features(x_train, 6)
x7 = add_polynomial_features(x_train, 7)
x8 = add_polynomial_features(x_train, 8)
x9 = add_polynomial_features(x_train, 9)

x2_test = add_polynomial_features(x_test, 2)
x3_test = add_polynomial_features(x_test, 3)
x4_test = add_polynomial_features(x_test, 4)
x5_test = add_polynomial_features(x_test, 5)
x6_test = add_polynomial_features(x_test, 6)
x7_test = add_polynomial_features(x_test, 7)
x8_test = add_polynomial_features(x_test, 8)
x9_test = add_polynomial_features(x_test, 9)

mylr2 = MyLinearRegression([[88.85],[-9.0 ], [0]])
mylr3 = MyLinearRegression([[88.85],[-9.0 ], [0], [0]])
mylr4 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0]])

mylr5 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0], [0]])
mylr6 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0], [0], [0]])

mylr7 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0], [0], [0], [0]])
mylr8 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0], [0], [0], [0], [0]])

mylr9 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0], [0], [0], [0], [0], [0]])

mylr2.fit_(x2, y_train)
mylr2.alpha = 0.00001
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.00003
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.0001
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.0003
mylr2.fit_(x2, y_train)
y_n.append(mylr2.cost_(x2_test,y_test))

mylr3.fit_(x3, y_train)
mylr3.alpha = 0.00001
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.00003
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.0001
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.0003
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.001
mylr3.fit_(x3, y_train)
y_n.append(mylr3.cost_(x3_test,y_test))

mylr4.fit_(x4, y_train)
mylr4.alpha = 0.00001
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.00003
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.0001
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.0003
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.001
mylr4.fit_(x4, y_train)
y_n.append(mylr4.cost_(x4_test,y_test))

mylr5.fit_(x5, y_train)
mylr5.alpha = 0.00001
mylr5.fit_(x5, y_train)
mylr5.alpha = 0.00003
mylr5.fit_(x5, y_train)
mylr5.alpha = 0.0001
mylr5.fit_(x5, y_train)
y_n.append(mylr5.cost_(x5_test,y_test))

mylr6.fit_(x6, y_train)
mylr6.alpha = 0.00001
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.00003
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.0001
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.0003
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.001
mylr6.fit_(x6, y_train)
y_n.append(mylr6.cost_(x6_test,y_test))

mylr7.fit_(x7, y_train)
mylr7.alpha = 0.00001
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.00003
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.0001
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.0003
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.001
mylr7.fit_(x7, y_train)
y_n.append(mylr7.cost_(x7_test,y_test))

mylr8.fit_(x8, y_train)
mylr8.alpha = 0.000003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.00001
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.00003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.0001
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.0003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.001
mylr8.fit_(x8, y_train)
y_n.append(mylr8.cost_(x8_test,y_test))

mylr9.alpha = 0.00000001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0000001
mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.0000003
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.0000008
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.000001
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.000003
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.000008
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.00001
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.00003
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.00008
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.0001
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.0003
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.0008
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.001
# mylr9.fit_(x9, y_train)
# mylr9.alpha = 0.003
# mylr9.fit_(x9, y_train)
y_n.append(mylr9.cost_(x9_test,y_test))
print(mylr9.thetas)
print(mylr9.cost_(x9_test,y_test))
plt.bar(range(2, 10), y_n)

plt.xlabel('n exponent')
plt.ylabel('cost')
plt.title('train nine separate Linear Regression models with polynomial hypotheses with degrees ranging from 2 to 10.')
plt.show()
