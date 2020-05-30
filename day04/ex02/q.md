**1 - What is overﬁtting?**

- 모집단이 아닌 train데이터에만 cost함수가 최적으로 맞춰진것이다. 이 경우 실제 새로운 데이터값이 들어오면 오차가 커질 수 있다.

**2 - What do you think underﬁtting might be?**

- **overfitting**이 다항식이 복잡해져서 일어났다면 **underfitting**은 다항식이 너무 단순하여 오차가 줄어드지 않는 경우를 말할 것 같습니다.



**3 - Why is it important to split the data set in a training and a test set?**

- 새로운 데이터를 테스트해보아서 **overfitting**을 막기 위함입니다.



**4 - If a model overﬁts, what will happen when you compare its performance on the training set and the test set?**

- training set에서는 최고의 효율로 예측하겠지만 test set에서는 오차가 많이 발생합니다.

**5 - If a model underﬁts, what do you think will happen when you compare its performance on the training set and the test set?**

- underfitting이 일어난다면 test set과 training set 모두 오차가 많이 발생할 것 같습니다.