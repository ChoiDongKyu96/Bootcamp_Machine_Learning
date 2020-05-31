# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_spliter.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/28 15:08:21 by dochoi            #+#    #+#              #
#    Updated: 2020/05/31 21:47:12 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the traning set., →
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1. proportion: has to be a float,
    the proportion of the dataset that will be assigned to the training set., →
    Returns: (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray None if x or y is an empty numpy.ndarray.
    None if x and y do not share compatible dimensions.
    Raises: This function should not raise any Exception. """
    n = int(float(x.shape[0] * proportion))
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    x_train = x[0:n]
    x_test = x[n:]
    y_train = y[0:n]
    y_test = y[n:]
    return (x_train, x_test, y_train, y_test)