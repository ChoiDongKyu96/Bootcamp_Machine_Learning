# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_log_gradient.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/29 17:26:49 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 02:46:45 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def add_intercept(x):
    if len(x) == 0 or x.ndim >= 3:
        return None
    if x.ndim == 1:
        return np.vstack((np.ones(len(x)), x)).T
    else:
        return np.insert(x, 0, 1, axis=1)

def logistic_predict_(x, theta):

    if (len(x) == 0 or len(theta) == 0 or x.shape[1] != (len(theta) - 1)):
        return None
    return 1 / (1 + pow(math.e, -add_intercept(x) @ theta))

def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible dimensions.,
     Args:
     x: has to be an numpy.ndarray, a matrix of dimension m * n.
     y: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector (n +1) * 1.
     Returns: The gradient as a numpy.ndarray, a vector of dimension n * 1,
     containg the result of the formula for all j.,
     None if x, y, or theta are empty numpy.ndarray.
     None if x, y and theta do not have compatible dimensions.
     Raises: This function should not raise any Exception. """

    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    return add_intercept(x).T @ (logistic_predict_(x, theta) - y) / len(x)

y1 = np.array([1]).reshape(-1,1)
x1 = np.array([4]).reshape(-1,1)
theta1 = np.array([[2], [0.5]])
print(vec_log_gradient(x1, y1, theta1))

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(vec_log_gradient(x2, y2, theta2))

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(vec_log_gradient(x3, y3, theta3))