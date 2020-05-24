# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/23 14:31:41 by dochoi            #+#    #+#              #
#    Updated: 2020/05/24 18:58:57 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def fit_(x, y, theta, alpha, max_iter):
    """ Description: Fits the model to the training dataset contained in x and y.
    Args:
    x:
    has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y:
    has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises: This function should not raise any Exception. """
    def gradient(x, y, theta):
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
        return (add_intercept(x).T @ (predict_(x, theta).reshape(-1,1) - y)) / len(x)


    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if x.shape != y.shape:
        return None
    theta_r = np.array([theta[0], theta[1]], dtype=float)
    while max_iter:
        for i, v in enumerate(gradient(x, y, theta_r)):
            theta_r[i] -= (alpha * v)
        max_iter -= 1
    return theta_r

def add_intercept_(x):
    if len(x) == 0 or x.ndim >= 3:
        return None
    if x.ndim == 1:
        return np.vstack((np.ones(len(x)), x)).T
    else:
        return np.insert(x, 0, 1, axis=1)

def predict(x, theta):
    if len(theta) != 2 or len(x) == 0:
        return None
    return add_intercept_(x) @ theta
x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1])
# Example 0:
theta1 = fit_(x, y, theta,alpha=5e-8, max_iter = 1500000)
print(theta1)
# Output: array([[1.40709365], [1.1150909 ]])
# Example 1:
print(predict(x, theta1))
 # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])