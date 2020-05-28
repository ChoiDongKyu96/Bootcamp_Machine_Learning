# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/28 17:35:30 by dochoi            #+#    #+#              #
#    Updated: 2020/05/29 01:49:01 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math
import numpy as np

def sigmoid_(x):
    """ Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.ndarray, a vector
    Returns: The sigmoid value as a numpy.ndarray.
    None if x is an empty numpy.ndarray.
    Raises: This function should not raise any Exception. """
    if len(x) == 0:
        return None
    def func(x):
        return 1 / (1 + pow(math.e, -x))
    return np.vectorize(func)(x)


x = np.array([-4])
print(sigmoid_(x))
x = np.array([2])
print(sigmoid_(x))
x = np.array([[-4], [2], [0]])
print(sigmoid_(x))