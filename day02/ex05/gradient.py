# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/26 16:31:37 by dochoi            #+#    #+#              #
#    Updated: 2020/05/26 16:50:05 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception. """


    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)
    def predict_(x, theta):
        if x.ndim == 1:
            x = x[:,np.newaxis]
        if len(x) == 0:
            return None
        return x @ theta

    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    return x.T @ (predict_(x, theta) - y) / len(x)

x = np.array([ [ -6, -7, -9], [ 13, -2, 14],[ -7, 14, -1], [ -8, -4, 6], [ -5, -9, 6], [ 1, -5, 11], [ 9, -11, 8]])

y = np.array([2, 14, -13, 5, 12, 4, -19])
theta1 = np.array([3,0.5,-6])
# Example :
print(gradient(x, y, theta1))
## Output: array([ -37.35714286, 183.14285714, -393. ])
theta2 = np.array([0,0,0])
print(gradient(x, y, theta2)) # Output: array([ 0.85714286, 23.28571429, -26.42857143])
