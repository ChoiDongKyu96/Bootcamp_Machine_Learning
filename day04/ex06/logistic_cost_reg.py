# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logistic_cost_reg.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/31 03:17:13 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 03:19:25 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def vec_log_loss_(y, y_hat, eps=1e-15):
    if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):
        return None
    y_hat = y_hat - eps
    return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ) / len(y_hat)

def l2(theta):
    if len(theta) == 0:
        return None
    l2_lst = theta * theta
    return np.sum(l2_lst) - l2_lst[0]

def reg_log_cost_(y, y_hat, theta, lambda_):
    """Computes the regularized cost of a logistic regression model from two non-empty
    ,! numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized cost as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (len(y) == 0 or len(y_hat) == 0 or len(theta) == 0):
        return None
    if (y.shape != y_hat.shape):
        return None
    return vec_log_loss_(y, y_hat)  + (lambda_ * l2(theta)/(2 * len(y_hat)))

y = np.array([1, 1, 0, 0, 1, 1, 0])
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
theta = np.array([1, 2.5, 1.5, -0.9])
# Example :
print(reg_log_cost_(y, y_hat, theta, .5))
# Output:0.43377043716475955
# Example :
print(reg_log_cost_(y, y_hat, theta, .05))
# Output:0.13452043716475953
# Example :
print(reg_log_cost_(y, y_hat, theta, .9))
# Output:0.6997704371647596