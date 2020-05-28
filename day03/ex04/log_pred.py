# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_pred.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/29 01:33:57 by dochoi            #+#    #+#              #
#    Updated: 2020/05/29 02:02:06 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
Args:
    x:
        has to be an numpy.ndarray, a vector of dimension m * n.
    theta:
        has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
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

x = np.array([4]).reshape(-1, 1)
theta = np.array([[2], [0.5]])
print(logistic_predict_(x, theta))

x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(logistic_predict_(x2, theta2))

x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(logistic_predict_(x3, theta3))