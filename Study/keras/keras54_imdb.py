from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

# 실습 시작!! 완성하시오!!

# print(x_train[0], type(x_train[0]))
# print(x_train[1], type(x_train[1]))
# <class 'list'>

# print(y_train[0])
# print(x_train.shape)
# print(type(x_train))

# print("최대길이 : ", max(len(i) for i in x_train))
# print("최대길이 : ", sum(map(len, x_train))/ len(x_train))

# print("최대길이 : ", max(len(i) for i in x_test))

#@ 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=2494, padding='pre') 
x_test = pad_sequences(x_test, maxlen=2494, padding='pre')  

print(x_train.shape, x_test.shape) # (25000, 2494), (25000, 2494)
print(type(x_train), type(x_train[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

#@ y확인
print(np.unique(y_train)) # [0, 1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (25000, 2), (25000, 2)

print(np.unique(x_test))

#@ 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation="softmax"))

#@ 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=200)

#@ 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)
'''
acc :  0.8196799755096436
'''