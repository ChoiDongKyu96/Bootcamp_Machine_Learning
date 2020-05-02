# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cost.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/01 20:49:08 by dochoi            #+#    #+#              #
#    Updated: 2020/05/01 22:24:37 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt
import numpy as np

def cost_elem_(y, y_hat):
    """ Description: Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
    Args: y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns: J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    Raises: This function should not raise any Exception. """
    if y.ndim == 1:
        y = y[:,np.newaxis]
    if y.shape != y_hat.shape or len(y) == 0 or len(y_hat) ==0:
        return None
    return ((y - y_hat) ** 2) / (2 * len(y))

def cost_(y, y_hat):
    """ Description: Calculates the value of cost function.
    Args:
    y: has to be an numpy.ndarray, a vector. y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    J_value : has to be a float"""
    if y.ndim == 1:
        y = y[:,np.newaxis]
    if y_hat.ndim == 1:
        y_hat = y_hat[:,np.newaxis]
    if y.shape != y_hat.shape or len(y) == 0 or len(y_hat) ==0:
        return None
    return np.squeeze(sum((y - y_hat) ** 2 / (2 * len(y))))

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
     x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
     Returns: y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
    Raises: This function should not raise any Exceptions. """
    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)
    if x.ndim == 1:
        x = x[:,np.newaxis]
    if len(theta) - 1 != x.shape[1]  or len(x) == 0:
        return None
    return add_intercept(x) @ theta



x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(cost_elem_(y1, y_hat1))
print(cost_(y1, y_hat1))
x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])
print(cost_elem_(y2, y_hat2))
print(cost_(y2, y_hat2))

x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])
print(cost_elem_(y3, y_hat3))
print(cost_(y3, y_hat3))
print(cost_(y3, y3))