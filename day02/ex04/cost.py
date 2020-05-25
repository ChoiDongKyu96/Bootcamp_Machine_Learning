# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cost.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/26 01:47:30 by dochoi            #+#    #+#              #
#    Updated: 2020/05/26 01:48:32 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def cost_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.ndarray, without any for loop.
    The two arrays must have the same dimensions.,
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat:
    has to be an numpy.ndarray, a vector.
    Returns:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.ndarray. None if y and y_hat does not share the same dimensions.
    Raises: This function should not raise any Exceptions. """
    if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
        return None
    return ((y_hat - y) @ (y_hat - y)) / (2 * len(y))

import numpy as np
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(cost_(X,Y))