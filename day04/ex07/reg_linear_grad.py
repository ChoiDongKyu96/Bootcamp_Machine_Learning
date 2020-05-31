# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_linear_grad.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/31 13:37:39 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 13:58:34 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

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
    if len(theta) - 1 != x.shape[1]  or len(x) == 0:
        return None
    return add_intercept(x) @ theta

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two
    for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all
    ,! j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if len(y) == 0 or len(x) == 0 or len(theta) == 0:
        return None
    if len(y) != len(x) or len(x[0]) != len(theta) - 1:
        return None
    y_hat = predict_(x, theta)
    answer = np.sum(y_hat- y)
    for j in range(0, len(theta) - 1):
        temp = 0.0
        for i in range(len(x)):
            temp += ((y_hat[i] - y[i]) * x[i][j])
        answer = np.append(answer, temp + lambda_ * theta[j + 1])
    return (answer / len(x)).reshape(-1,1)

def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, without any
    for-loop. The three arrays must have compatible dimensions.
    Args:
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if len(y) == 0 or len(x) == 0 or len(theta) == 0:
        return None
    if len(y) != len(x) or len(x[0]) != len(theta) - 1:
        return None
    answer = add_intercept(x).T @ (predict_(x, theta) - y) + (lambda_ * theta)
    answer[0] -= lambda_ * theta[0]
    return answer / len(x)

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])

# Example 1.1:
print(reg_linear_grad(y, x, theta, 1))
# Output:
# Example 1.2:
print(vec_reg_linear_grad(y, x, theta, 1))
# # Output:
# # Example 2.1:
print(reg_linear_grad(y, x, theta, 0.5))
# # Output:

# # Example 2.2:
print(vec_reg_linear_grad(y, x, theta, 0.5))
# # Output:

# # Example 3.1:
print(reg_linear_grad(y, x, theta, 0.0))

# # Example 3.2:
print(vec_reg_linear_grad(y, x, theta, 0.0))
