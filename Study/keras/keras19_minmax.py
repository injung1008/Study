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


# print(datasets.feature_names)
# print(datasets.DESCR)


# # # 완료 하시오 
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=66)
# print(x.shape) #(506,13)
# print(x_train.shape) #(354,13)
# print(y.shape)

# # model 구성 
# model = Sequential()
# model.add(Dense(1, input_dim=13))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(123, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(1))



# model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=1)

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss )

# y_predict = model.predict(x_test)
# print('y_predict : ', y_predict)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print(r2)




