import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_csv('./_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

datasets = df.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]
# print(x.shape)
# print(y.shape)
# print(np.unique(y)) #7개 [3 4 5 6 7 8 9]


x_train, x_test, y_train, y_test =train_test_split(x,y,
    train_size = 0.7, shuffle = True, random_state=9)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() 
y_test = encoder.transform(y_test).toarray() 

print(y_test.shape) #(1470, 7)
print(x_test.shape) #(1470, 11)
print(x_train.shape) #(3428, 11)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape) #(105, 4)
# print(x_test.shape)  #(45, 4)
# print(y_test.shape)  #(45, 3)
# print(y_train.shape)  #(105, 3)
# print(y_test)

x_train = x_train.reshape(3428, 11, 1) 
x_test = x_test.reshape(1470, 11, 1) 




model = Sequential()
model.add(Conv1D(10,2, input_shape=(11,1)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam'
                         , metrics=['accuracy']) 

# #훈련

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)
model.compile(loss='mse',optimizer=optimizer,
                    metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
start_time = time.time()
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
#es는 조건이맞으면 끝나고, reduce_lr도 조건 맞으면 끝나지만, factor=0.5 로 해놓으면 감소가 없으면 러닝 레이트가 0.5 만큼 줄어든다.

hist = model.fit(x_train, y_train, epochs=300, verbose=1,validation_split=0.2,
batch_size=1, callbacks=[es, reduce_lr])


end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )

#  0.5823129415512085
#
