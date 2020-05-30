# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_loss.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/29 13:23:23 by dochoi            #+#    #+#              #
#    Updated: 2020/05/29 13:45:58 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math
def logistic_predict_(x, theta):
    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    if (len(x) == 0 or len(theta) == 0 or x.shape[1] != (len(theta) - 1)):
        return None
    return 1 / (1 + pow(math.e, -add_intercept(x) @ theta))

def log_loss_(y, y_hat, eps=1e-15):
    """ Computes the logistic loss value. Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Returns: The logistic loss value as a float. None on any error.
    Raises: This function should not raise any Exception. """
    if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):
        return None
    def func(x):
        return math.log(x)
    def func2(x):
        return (math.log(1 - x))
    y_hat = y_hat - eps
    y_hat_log = np.vectorize(func)(y_hat)
    y_hat_log_inv = np.vectorize(func2)(y_hat)
    return -sum(y * y_hat_log + (1 - y) * y_hat_log_inv)  / len(y_hat)

y1 = np.array([1]).reshape(-1,1)
x1 = np.array([4]).reshape(-1,1)
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print(log_loss_(y1, y_hat1))

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(log_loss_(y2, y_hat2))

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print(log_loss_(y3, y_hat3))