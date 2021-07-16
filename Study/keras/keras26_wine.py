import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


#acc 0.8 이상
#다중 분류 사용 (y = 0,1,2)
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)
print(datasets)
# x = datasets.data
# y = datasets.target

# print(x.shape, y.shape) #(178, 13) (178,)
# print(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# #print(y) #[1. 0. 0.]

# x_train, x_test, y_train, y_test =train_test_split(x,y,
#     train_size = 0.7, shuffle = True, random_state=66)

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = Sequential()
# model.add(Dense(1, input_dim=13)) 
# model.add(Dense(100, activation='relu'))
# model.add(Dense(123, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(3, activation='softmax')) #y lable 3ea

# #훈련
# model.compile(loss='categorical_crossentropy',optimizer='adam',
#                     metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping

# es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)

# model.fit(x_train, y_train, epochs=100, verbose=1,
# batch_size=1, validation_split=0.2,  callbacks=[es])

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('metrix : ', loss[1] )

# accuracy: 0.8889
# loss :  [0.4022406041622162, 0.8888888955116272]
# metrix :  0.8888888955116272