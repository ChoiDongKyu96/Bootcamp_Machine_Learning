# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_model.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 13:12:25 by dochoi            #+#    #+#              #
#    Updated: 2020/05/27 14:02:08 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args: x: has to be an numpy.ndarray, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised., →
    Returns: The matrix of polynomial features as a numpy.ndarray,
        of dimension m * n, containg he polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises: This function should not raise any Exception. """
    temp = x.copy()
    for i in range(2, power + 1):
        temp = np.append(temp, np.power(x, i), axis=1 )
    return temp

x = np.arange(1,6).reshape(-1, 1)
# print(np.append(x,x,axis=1))
# print(x)
print(add_polynomial_features(x, 3))
# # Output: array([[ 1, 1, 1], [ 2, 4, 8], [ 3, 9, 27], [ 4, 16, 64], [ 5, 25, 125]])
print(add_polynomial_features(x, 6))