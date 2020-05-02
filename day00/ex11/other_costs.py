# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    other_costs.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/02 19:42:58 by dochoi            #+#    #+#              #
#    Updated: 2020/05/02 20:09:15 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math

def mse_(y, y_hat):
    """ Description: Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises: This function should not raise any Exceptions. """
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    return ((y_hat - y) @ (y_hat - y)) / (len(y))

def rmse_(y, y_hat):
    """ Description: Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises: This function should not raise any Exceptions. """
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    return math.sqrt(((y_hat - y) @ (y_hat - y)) / (len(y)))

def mae_(y, y_hat):
    """ Description: Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
    mae: has to be a float. None if there is a matching dimension problem.
    Raises: This function should not raise any Exceptions. """
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    return (sum(abs(y_hat - y))) / (len(y))

def r2score_(y, y_hat):
    """ Description: Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns: r2score: has to be a float.
    None if there is a matching dimension problem. Raises:
    This function should not raise any Exceptions. """
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    y_m = sum(y) / len(y)
    return 1 - (((y_hat - y) @ (y_hat - y)) / ((y_hat - y_m) @ (y_hat - y_m)))

# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from math import sqrt

# x = np.array([0, 15, -9, 7, 12, 3, -21])
# y = np.array([2, 14, -13, 5, 12, 4, -19])
# print(mse_(x,y))
# print(mean_squared_error(x,y))
# print(rmse_(x,y))
# print(sqrt(mean_squared_error(x,y)))
# print(mae_(x,y))
# print(mean_absolute_error(x,y))
# print(r2score_(x,y))
# print(r2_score(x,y))
