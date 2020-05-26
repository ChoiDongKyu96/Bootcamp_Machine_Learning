# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/26 16:52:51 by dochoi            #+#    #+#              #
#    Updated: 2020/05/27 01:29:42 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np

def fit_(x, y, theta, alpha, n_cycle):
    """ Description: Fits the model to the training dataset contained in x and y.
    Args:
    x:
    has to be a numpy.ndarray, a vector of dimension m * n:  (number of training examples, number of features).
    y:
    has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
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
            if x.ndim == 1:
                x = x[:,np.newaxis]
            if len(x) == 0:
                return None
            return add_intercept(x) @ theta
        if len(x) == 0 or len(y) == 0 or len(theta) == 0:
            return None
        return add_intercept(x).T @ (predict_(x, theta).reshape(-1,1) - y) / len(x)


    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    theta_r = np.array(theta, dtype=float)
    while n_cycle:
        for i, v in enumerate(gradient(x, y, theta_r)):
            theta_r[i] -= (alpha * v)
        n_cycle -= 1
    return theta_r

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
    return add_intercept(x) @ theta


X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
# Example 0:
theta2 = fit_(X2, Y2, theta2, alpha = 0.0005, n_cycle=42000)
print(theta2) # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])
# # Example 1:
print(predict_(X2, theta2)) # Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])