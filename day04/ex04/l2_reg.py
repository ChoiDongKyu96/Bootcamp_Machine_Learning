# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    l2_reg.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/31 02:37:01 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 02:46:58 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if len(theta) == 0:
        return None
    l2_lst = [i * i for i in theta]
    return sum(l2_lst) - l2_lst[0]


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function
    """
    if len(theta) == 0:
        return None
    l2_lst = theta * theta
    return np.sum(l2_lst) - l2_lst[0]

x = np.array([2, 14, -13, 5, 12, 4, -19])
# Example 1:
print(iterative_l2(x))
# Output:911.0
# Example 2:
print(l2(x))
# Output:911.0
y = np.array([3,0.5,-6])
# Example 3:
print(iterative_l2(y))
# Output:36.25
# Example 4:
print(l2(y))
# Output: