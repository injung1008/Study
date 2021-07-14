#보스톤으로 min max

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # 3행 10열 
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)

#print(np.min(x), np.max(x)) # 총 데이터에서 최솟값 =0.0 최댓값 = 711.0

#데이터 전처리 (최댓값으로 x를나눠준다 - min-max- scaler방법)
# x = x/711.
# x = x/np.max(x)
#x = (x - np.min(x))/ (np.max(x) - np.min(x)) #전체 데이터 정규화
x_train, x_test, y_train, y_test = train_test_split(x, 
                        y, train_size = 0.7, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler #min max 정규화 import 
scaler = MinMaxScaler() 
# print('x_train:', x_train)
# print('x_test:', x_test)

#fit = (전처리) 무언가를 훈련,실행 시키다의 개념 공식을 만들어준다 
scaler.fit(x_train)

# 실행 시킨 결과를 변환시 키는 과정 (배출의 과정)
x_train = scaler.transform(x_train)
print('x = ',x_train)
#스케일릴ㅇ 비율에 맞춰서 트랜스폼 된것이다 
#x_test = scaler.transform(x_test)


