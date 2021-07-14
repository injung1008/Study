#from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#MaxAbsScaler : 최대 절대값과 0이 각각 1, 0 이 되도록 스케일링 
# 절대값이 0-1사이에 매핑되도록 한다. 즉 -1 -1 사이로 재조정한다. 
# 양수 데이터로만 구성된 특정 데이터셋에서는 min-max와 유사하게 동작되며 큰이상치에 민감할 수 있다

#RobustScaler :평균과 분산대신 중앙값과 IQR(interquatile range) 사용, 아웃라이어의 영향을 최소화 (이상치에 민감하지 않다)
# IQR(Q3 -Q1) -데이터 범위를 나눈 구간으로 25%- 75% 구간을 말하며 표준정규분포에 비해 표준화후 동일한값을 더 넓게 분포하고 있다 

#QuantileTransformer = 1000개의 분위를 사용하여 데이터를 균등분포 시킨다 (이상치에 민감하지 않음) 
# 0-1사이로 압축, 평균이 0 표준편차1이지만 정규분포를 따르지 않는 분포이다 

#PowerTransformer = 데이터의 특성별로 정규분포 형태에 가깝도록 변환

#보스톤으로 min max

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x)) # 총 데이터에서 최솟값 =0.0 최댓값 = 711.0

#데이터 전처리 (최댓값으로 x를나눠준다 - min-max- scaler방법)
x = x/711.
x = x/np.max(x)


print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size = 0.7, shuffle=True, random_state=66)
# print(x.shape) #(506,13)
# print(x_train.shape) #(354,13)
# print(y.shape)
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# model 구성 
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(123, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)


# MaxAbsScaler - r2= 0.859907857724007
# RobustScaler - r2= 0.42304845151189163
# QuantileTransformer - r2 = 0.8406405438316127
# PowerTransformer - r2 =0.8473388112358081
