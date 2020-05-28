# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_train.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 13:38:10 by dochoi            #+#    #+#              #
#    Updated: 2020/05/27 17:58:05 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=2e-5   , n_cycle=1300000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def add_intercept(self,x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    def gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        return self.add_intercept(x).T @ (self.predict_(x).reshape(-1,1) - y) / len(x)

    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        n_cycle= self.n_cycle
        while n_cycle:
            for i, v in enumerate(self.gradient(x, y)):
                self.thetas[i] -= (self.alpha * v)
            n_cycle -= 1
        return self.thetas

    def predict_(self,x):
        if x.ndim == 1:
            x = x[:,np.newaxis]
        if len(self.thetas) - 1 != x.shape[1]  or len(x) == 0:
            return None
        return self.add_intercept(x) @ self.thetas

    def cost_elem_(self, x, y):
        y_hat = self.predict_(x)
        if y.ndim == 1:
            y = y[:,np.newaxis]
        if y.shape != y_hat.shape or len(y) == 0 or len(y_hat) ==0:
            return None
        return ((y - y_hat) ** 2) / (2 * len(y))

    def cost_(self, x, y):
        y_hat = self.predict_(x)
        if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
            return None
        return (sum((y_hat - y) * (y_hat - y)).squeeze() / (2 * len(y)))


def add_polynomial_features(x, power):
    temp = x.copy()
    for i in range(2, power + 1):
        temp = np.append(temp, np.power(x, i), axis=1 )
    return temp

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.,
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns: x' as a numpy.ndarray. None if x is a non-empty numpy.ndarray.
    Raises: This function shouldn't raise any Exception. """
    pivot = max(x) - min(x)
    return (x - min(x)) / pivot

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args: x:
    has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception. """
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
x = minmax(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = minmax(y)

x2 = add_polynomial_features(x, 2)
x3 = add_polynomial_features(x, 3)
x4 = add_polynomial_features(x, 4)
x5 = add_polynomial_features(x, 5)
x6 = add_polynomial_features(x, 6)
x7 = add_polynomial_features(x, 7)
x8 = add_polynomial_features(x, 8)
x9 = add_polynomial_features(x, 9)

mylr2 = MyLinearRegression([[88.85],[-9.0 ], [1]])
mylr3 = MyLinearRegression([[88.85],[-9.0 ], [1], [1]])
mylr4 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1]])

mylr5 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1]])
mylr6 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1]])

mylr7 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1], [1]])
mylr8 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1], [1], [1]])

mylr9 = MyLinearRegression([[ 11.62659002],
 [-24.78318432],
 [ -3.92454659],
 [  0.48946634],
 [  2.36855332],
 [  3.20499866],
 [  3.58663679],
 [  3.75580837],
 [  3.81876051],
 [  3.82568518]])
# mylr2.fit_(x2, y)
# y_n.append(mylr2.cost_(x2,y))

# mylr3.fit_(x3, y)
# y_n.append(mylr3.cost_(x3,y))

# mylr4.fit_(x4, y)
# y_n.append(mylr4.cost_(x4,y))

# mylr5.fit_(x5, y)
# y_n.append(mylr5.cost_(x5,y))

# mylr6.fit_(x6, y)
# y_n.append(mylr6.cost_(x6,y))

# mylr7.fit_(x7, y)
# y_n.append(mylr7.cost_(x7,y))

# mylr8.fit_(x8, y)
# y_n.append(mylr8.cost_(x8,y))

mylr9.fit_(x9, y)
y_n.append(mylr9.cost_(x9,y))
print(mylr9.thetas)
print(mylr9.cost_(x9,y))
# plt.bar(range(2, 10), y_n)

# plt.xlabel('n exponent')
# plt.ylabel('cost')
# plt.title('train nine separate Linear Regression models with polynomial hypotheses with degrees ranging from 2 to 10.')
# plt.show()
