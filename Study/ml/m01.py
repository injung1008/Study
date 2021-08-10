# 다중분류 ( y가 0,1,2로 이루어져 있을때)
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150,4) (150,)
print(y) # y = 0,1,2

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66,stratify=y)


#! scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#!model 구성 
from sklearn.svm import LinearSVC
model = LinearSVC() #LinearSVC 모델을 만든다 / 컴파일도 포함되어 있다 
model.fit(x_train, y_train) 

# ValueError: y should be a 1d array, got an array of shape (105, 3) instead.
#! -> 사이킥런의 대부분은 원핫인코딩 할필요가 없다 svc에서 원핫인코딩까지 해주겠다

#사이킥런은 평가하는것이 기존 evaluate -> score이다  
results = model.score(x_test, y_test)
print(results)
# acc = 0.9333333333333333

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("auuracy_score : ", acc)


print('==========예측=============')
print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2) #[0 1 1 0 2]




