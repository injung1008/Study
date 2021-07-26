from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd 
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, test_split=0.2
)

#리스트는 각각 의 크기가 다를 수 있다 그러므로 훈련데이터의 0번과 1번의 길이가 다를 수 있다 
#print(x_train[0], type(x_train[0]))
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 
# 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19,
#  102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 
#  7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 
#  83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 
#  1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 
#  44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 
#  29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>
#print(x_train[1], type(x_train[0]))
print(y_train[0]) #3
print(len(x_train[0]),len(x_train[1])) #87 56
# -> 글자 크기가 다름 패딩 해줘야함 
#shpae는 넘파이의 기능이기때문에 list는 사용불가하다 
#print(x_train[0].shape)

#리스트로 이루어진 넘파이다 하지만 각 요소는 리스트로 되어있다 
#print(x_train.shape,x_test.shape) #(8982,) (2246,)
#print(y_train.shape,y_test.shape) #(8982,) (2246,)
#print(type(x_train)) #<class 'numpy.ndarray'>

#각각의 리스트는 길이가 다르기때문에 길이를 확인하고 최대 길이로 전체를 0을 채워서 맞춰줘야한다
# !print("뉴스 기사의 최대길이 : ", max(list(map(len, x_train)))) #  2376
#print(list(map(len, x_train)))
#! print("뉴스 기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #145.5398574927633

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
# print(x_train[0])
# print(type(x_train), type(x_train[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

#! y data 카테고리화 시켜줘야한다 
from tensorflow.keras.utils import to_categorical
print(np.unique(y_train)) #총 46개
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 
# 13 14 15 16 17 18 19 20 21 22 23        
#  24 25 26 27 28 29 30 31 32 33 34 35 36 
# 37 38 39 40 41 42 43 44 45]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델 구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=11, input_length=100))
model.add(LSTM(532))
model.add(Dense(346, activation='relu'))
model.add(Dense(246, activation='relu'))
model.add(Dense(146))
model.add(Dense(46, activation='softmax'))
model.summary()

#3. 컴파일, 훈련 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1000)

#평가 예측 
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)
