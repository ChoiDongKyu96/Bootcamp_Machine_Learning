# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    other_metrics.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 18:09:56 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 18:46:33 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def accuracy_score_(y, y_hat):
    """ Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels y_hat:a numpy.ndarray for the predicted labels
    Returns: The accuracy score as a float. None on any error.
    Raises: This function should not raise any Exception. """
    return np.count_nonzero(y==y_hat) / float(len(y))

def precision_score_(y, y_hat, pos_label=1):
    """ Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int,
    the class on which to report the precision_score (default=1)
    Returns: The precision score as a float.
    None on any error.
    Raises: This function should not raise any Exception. """
    if np.count_nonzero(y_hat == pos_label) == 0:
        return 0
    return np.count_nonzero((y_hat == y) & (y_hat == pos_label)) / float(np.count_nonzero(y_hat==pos_label))

def recall_score_(y, y_hat, pos_label=1):
    """ Compute the recall score.
    Args: y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: The recall score as a float. None on any error.
    Raises: This function should not raise any Exception. """

    return  np.count_nonzero((y_hat == y) & (y_hat == pos_label)) / float(np.count_nonzero((y_hat == y) & (y_hat == pos_label)) + np.count_nonzero((y_hat != y) & (y_hat != pos_label)))

def f1_score_(y, y_hat, pos_label=1):
    """ Compute the f1 score. Args: y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: The f1 score as a float. None on any error.
    Raises: This function should not raise any Exception"""
    pre =precision_score_(y,y_hat, pos_label=pos_label)
    re = recall_score_(y,y_hat, pos_label=pos_label)
    return 2 * pre * re / (pre + re)

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Example 1:
y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
# Accuracy ##
# your implementation
print(accuracy_score_(y, y_hat)) ## Output: 0.5 ##
# sklearn implementation
print( accuracy_score(y, y_hat))
 ## Output: 0.5

 #Precision ## your implementation
print(precision_score_(y, y_hat))
  ## Output: 0.4 ## sklearn implementation
print(precision_score(y, y_hat))
 ## Output: 0.4

0.4
# Recall ## your implementation
print(recall_score_(y, y_hat))
## Output: 0.6666666666666666 ## sklearn implementation
print(recall_score(y, y_hat))
 ## Output: 0.666666666666666

# F1-score ## your implementation
print(f1_score_(y, y_hat))
## Output: 0.6666666666666665
# ## sklearn implementation
print(f1_score(y, y_hat))
## Output: 0.666666666666666

# Example 2:
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## Output:0.625
## sklearn implementation
print(accuracy_score(y, y_hat))
## Output:0.625
# Precision
## your implementation
print(precision_score_(y, y_hat, pos_label='dog'))
## Output:0.6
## sklearn implementation
print(precision_score(y, y_hat, pos_label='dog'))
## Output:0.6
# Recall
## your implementation
print(recall_score_(y, y_hat, pos_label='dog'))
## Output:0.75
## sklearn implementation
print(recall_score(y, y_hat, pos_label='dog'))
## Output:0.75
# F1-score
## your implementation
print(f1_score_(y, y_hat, pos_label='dog'))
## Output:0.6666666666666665
## sklearn implementation
print(f1_score(y, y_hat, pos_label='dog'))
## Output:0.6666666666666665

# Example 3:
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
## Output:0.625
## sklearn implementation
print(accuracy_score(y, y_hat))
## Output:0.625
# Precision
## your implementation
print(precision_score_(y, y_hat, pos_label='norminet'))
## Output:0.6666666666666666
## sklearn implementation
print(precision_score(y, y_hat, pos_label='norminet'))
## Output:0.6666666666666666
# Recall
## your implementation
print(recall_score_(y, y_hat, pos_label='norminet'))
## Output:0.5
## sklearn implementation
print(recall_score(y, y_hat, pos_label='norminet'))
## Output:0.5
# F1-score
## your implementation
print(f1_score_(y, y_hat, pos_label='norminet'))
## Output:0.5714285714285715
## sklearn implementation
print(f1_score(y, y_hat, pos_label='norminet'))
## Output:0.5714285714285715
