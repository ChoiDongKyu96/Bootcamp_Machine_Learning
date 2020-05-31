# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_reg_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:56:12 by dochoi            #+#    #+#              #
#    Updated: 2020/06/01 01:55:02 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features

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
    if (pre * re) == 0:
        return 0
    return 2 * pre * re / (pre + re)

def zscore(x):
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = np.sqrt(var/ (len(x) - 1))
    return (x - mu) / std

csv_data_x = pd.read_csv("../resources/solar_system_census.csv")

csv_data_y = pd.read_csv("../resources/solar_system_census_planets.csv")

x = np.array(csv_data_x[["height","weight","bone_density"]])
y =  np.array(csv_data_y["Origin"]).reshape(-1,1)

x = zscore(x)


temp = data_spliter(x, y, 0.5)
x_train = temp[0]
x_test = temp[1]
y_train = temp[2]
y_test = temp[3]

x_test_add_poly = add_polynomial_features(x_test, 3) # degree를 3으로 한다.
x_train_add_poly = add_polynomial_features(x_train, 3)
x = add_polynomial_features(x,3)

y_train0 = np.array([1 if i == 0 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test0 = np.array([1 if i == 0 else 0 for i in y_test]).reshape(-1,1)

y_train1 = np.array([1 if i == 1 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test1 = np.array([1 if i == 1 else 0 for i in y_test]).reshape(-1,1)

y_train2 = np.array([1 if i == 2 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test2 = np.array([1 if i == 2 else 0 for i in y_test]).reshape(-1,1)

y_train3 = np.array([1 if i == 3 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
y_test3 = np.array([1 if i == 3 else 0 for i in y_test]).reshape(-1,1)
theta = np.array([[1],
    [1],
    [ 1],
    [1],[1],[ 1],[ 1],[ 1],[ 1],[ 1]], dtype=float)
mylr0 = MyLR(theta, lambda_=0) # The ﬂying cities of Venus (0)
mylr1 = MyLR(theta, lambda_=0) #  United Nations of Earth (1)
mylr2 = MyLR(theta, lambda_=0) # Mars Republic (2)
mylr3 = MyLR(theta, lambda_=0) # The Asteroids’ Belt colonies (3).
y_n = []
y_n2= []
for i in range(10):
    mylr0.thetas = np.array(  [[-0.38004857],
    [ 0.12257596],
    [-1.13496089],
    [ 0.64144711],
    [ 0.13721429],
    [-0.46771826],
    [-1.18485222],
    [-0.46742162],
    [ 0.03928006],
    [-0.1718098 ]])
    mylr0.fit_(x_train_add_poly, y_train0)
    mylr0.alpha = 0.00003
    mylr0.fit_(x_train_add_poly, y_train0)
    mylr0.alpha = 0.00007
    mylr0.fit_(x_train_add_poly, y_train0)
    mylr0.alpha = 0.0001
    mylr0.fit_(x_train_add_poly, y_train0)
    mylr0.lambda_ += 0.1

    mylr1.thetas = np.array( [[-0.79899142],
    [-0.3785926 ],
    [ 1.24131593],
    [ 1.13327427],
    [-0.73841759],
    [-0.79814797],
    [ 0.03383971],
    [-0.40336806],
    [-0.76538218],
    [ 0.75970411],])

    mylr1.fit_(x_train_add_poly, y_train1)
    mylr1.alpha = 0.00003
    mylr1.fit_(x_train_add_poly, y_train1)
    mylr1.alpha = 0.00007
    mylr1.fit_(x_train_add_poly, y_train1)
    mylr1.alpha = 0.0001
    mylr1.fit_(x_train_add_poly, y_train1)
    mylr1.lambda_ += 0.1

    mylr2.thetas = np.array([[-1.01133034],
    [ 0.43171802],
    [ 1.63911229],
    [ 0.20881501],
    [-1.05478397],
    [ 0.30739429],
    [-1.12331241],
    [ 0.09578607],
    [ 0.84076726],
    [ 0.25000609],])

    mylr2.fit_(x_train_add_poly, y_train2)
    mylr2.alpha = 0.00003
    mylr2.fit_(x_train_add_poly, y_train2)
    mylr2.alpha = 0.00007
    mylr2.fit_(x_train_add_poly, y_train2)
    mylr2.alpha = 0.0001
    mylr2.fit_(x_train_add_poly, y_train2)
    mylr2.lambda_ += 0.1

    mylr3.thetas = np.array([[-1.37659546],
    [ 0.88245098],
    [ 0.06049297],
    [-0.38618096],
    [-0.22685866],
    [-1.05910852],
    [-0.12425495],
    [ 0.76153196],
    [-0.61360801],
    [-1.65141989]])

    mylr3.fit_(x_train_add_poly, y_train3)
    mylr3.alpha = 0.00003
    mylr3.fit_(x_train_add_poly, y_train3)
    mylr3.alpha = 0.00007
    mylr3.fit_(x_train_add_poly, y_train3)
    mylr3.alpha = 0.0001
    mylr3.fit_(x_train_add_poly, y_train3)
    mylr3.lambda_ += 0.1

    print(mylr0.thetas)
    print(mylr1.thetas)
    print(mylr2.thetas)
    print(mylr3.thetas)
    # 모델 생성 완료
    # 테스트 데이터 예측
    y_hat0 = mylr0.predict_(x_test_add_poly)
    y_hat1 = mylr1.predict_(x_test_add_poly)
    y_hat2 = mylr2.predict_(x_test_add_poly)
    y_hat3 = mylr3.predict_(x_test_add_poly)


    y_hat_total = np.append(y_hat0, y_hat1, axis=1)
    y_hat_total = np.append(y_hat_total, y_hat2, axis=1)
    y_hat_total = np.append(y_hat_total, y_hat3, axis=1)

    y_hat_pre_all = np.array([])
    # 데이터 확률 최댓값을 기준으로 클래스 분류
    for i in range(len(y_hat_total)):
        y_hat_pre_all = np.append(y_hat_pre_all, np.argmax(y_hat_total[i]))

    y_hat_pre_all = y_hat_pre_all.reshape(-1,1)

    print(mylr0.cost_(x_test_add_poly,y_test0))
    print(mylr1.cost_(x_test_add_poly,y_test1))
    print(mylr2.cost_(x_test_add_poly,y_test2))
    print(mylr3.cost_(x_test_add_poly,y_test3))
    y_n.append(f1_score_(y_test, y_hat_pre_all))
    y_n2.append(accuracy_score_(y_test, y_hat_pre_all))
    # np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

    # 시각화
plt.bar(range(0,10), y_n ,color=['black', 'red', 'green', 'blue','black', 'red', 'green', 'blue','black', 'red'])


plt.xlabel('lambda 0, 0.1, 0.2 ,..., 0.9')
plt.xticks(range(0,10))
plt.ylabel('f_score')
plt.ylim()
plt.title('**f_score for lambda change*')
plt.show()
print(y_n)
plt.bar(range(0,10), y_n2 ,color=['black', 'red', 'green', 'blue','black', 'red', 'green', 'blue','black', 'red'])

print(y_n2)
plt.xlabel('lambda 0, 0.1, 0.2 ,..., 0.9')
plt.xticks(range(0,10))
plt.ylabel('accuracy_score_')
plt.ylim()
plt.title('**accuracy_score_ for lambda change*')
plt.show()
