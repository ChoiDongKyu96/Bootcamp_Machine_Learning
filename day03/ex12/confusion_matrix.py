# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    confusion_matrix.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 18:53:51 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 20:06:38 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None , df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y_true:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    labels: optional, a list of labels to index the matrix. This may be used to reorder or
    ,! select a subset of labels. (default=None)
    Returns:
    The confusion matrix as a numpy ndarray.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if df_option == True:
        y_actu = pd.Series(y_true)
        y_pred = pd.Series(y_hat)
        df_confusion = pd.crosstab(y_actu, y_pred)
        if labels != None:
            return df_confusion.loc[labels, labels]
        return df_confusion
    else :
        K = max(len(np.unique(y_hat)) , len(np.unique(y)) ) # Number of classes
        name_yhat, index_y_hat = np.unique(y_hat, return_inverse=True)
        name_y, index_y = np.unique(y, return_inverse=True)

        max_y = name_y

        dict = {}
        if(len(name_yhat) > len(max_y)):
            max_y = name_yhat
        for idx, name in enumerate(max_y,):
            dict[name] = idx
        result = np.zeros((K, K), dtype=int)
        for i in range(len(y)):
            result[dict[y[i]]][dict[y_hat[i]]] += 1
        if labels != None:
            result_labels = []
            for elem in max_y:
                if elem not in labels:
                    result_labels = np.append(result_labels, dict[elem])
            arr = np.delete(result, result_labels,axis=0)
            arr = np.delete(arr, result_labels,axis=1)
            return arr
        return result

import numpy as np
from sklearn.metrics import confusion_matrix
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
# Example 1:
## your implementation
print(confusion_matrix_(y, y_hat))

## sklearn implementation
print(confusion_matrix(y, y_hat))


# # Example 2:
# ## your implementation
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))


## sklearn implementation
print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

print(confusion_matrix_(y, y_hat,df_option=1))
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet'], df_option=1))