# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_cost_reg.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/31 02:48:04 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 03:12:11 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def cost_(y, y_hat):
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    return ((y_hat - y) @ (y_hat - y)) / (2 * len(y))

def l2(theta):
    if len(theta) == 0:
        return None
    l2_lst = theta * theta
    return np.sum(l2_lst) - l2_lst[0]

def reg_cost_(y, y_hat, theta, lambda_):
    """Computes the regularized cost of a linear regression model from two non-empty
    ,! numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized cost as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (len(y) == 0 or len(y_hat) == 0 or len(theta) == 0):
        return None
    if (y.shape != y_hat.shape):
        return None
    return cost_(y, y_hat)  + (lambda_ * l2(theta)/(2 * len(y_hat)))

y = np.array([2, 14, -13, 5, 12, 4, -19])
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
theta = np.array([1, 2.5, 1.5, -0.9])
# Example :
print(reg_cost_(y, y_hat, theta, .5))
# Output:0.8503571428571429
# Example :
print(reg_cost_(y, y_hat, theta, .05))
# Output:0.5511071428571429
# Example :
print(reg_cost_(y, y_hat, theta, .9))
# Output:1.116357142857143