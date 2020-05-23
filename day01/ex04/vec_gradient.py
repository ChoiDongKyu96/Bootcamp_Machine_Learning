# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/22 17:44:14 by dochoi            #+#    #+#              #
#    Updated: 2020/05/23 15:14:59 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
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
        if len(theta) != 2 or len(x) == 0:
            return None
        return add_intercept(x) @ theta

    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape != y.shape:
        return None
    return add_intercept(x).T @ (predict_(x, theta) - y) / len(x)

# x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
# y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta1 = np.array([2, 0.7])
print(gradient(x, y, theta1))
#Output: array([21.0342574, 587.36875564])
theta2 = np.array([1, -0.4])
print(gradient(x, y, theta2))