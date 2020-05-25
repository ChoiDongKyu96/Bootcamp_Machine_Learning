# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    z-score.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/25 15:05:20 by dochoi            #+#    #+#              #
#    Updated: 2020/05/25 15:58:24 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args: x:
    has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception. """
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = math.sqrt(var/ (len(x) - 1))
    return (x - mu) / std

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))
#  array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(Y))
#  array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])
