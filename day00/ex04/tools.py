# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tools.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/30 16:21:45 by dochoi            #+#    #+#              #
#    Updated: 2020/04/30 16:43:52 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
    X as a numpy.ndarray, a vector of dimension m * 2. None if x is not a numpy.ndarray.
     None if x is a empty numpy.ndarray.
      Raises: This function should not raise any Exception. """
    if len(x) == 0 or x.ndim >= 3:
        return None
    if x.ndim == 1:
        return np.vstack((np.ones(len(x)), x)).T
    else:
        return np.insert(x, 0, 1, axis=1)


# x = np.arange(1, 6, dtype=float)
# print(add_intercept(x))
# y = np.arange(1, 10, dtype=float).reshape((3,3))
# print(y)
# print(add_intercept(y))