# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_model_extended.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 22:46:01 by dochoi            #+#    #+#              #
#    Updated: 2020/06/01 00:06:06 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of
    ,! 1 up to the power given in argument.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    power: has to be an int, the power up to which the columns of matrix x are going to be
    ,! raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), containg the
    ,! polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    temp = x.copy()
    for i in range(2, power + 1):
        temp = np.append(temp, np.power(x, i), axis=1 )
    return temp
