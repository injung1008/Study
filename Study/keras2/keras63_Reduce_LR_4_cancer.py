
from tensorflow.keras.layers import Dense, LSTM, Conv1D,Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(569, 30) (569,)
# print(y[:20])
# #[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y)) #[0 1] 중복되는게 없는 유니크한 값

#실습 : 모델 시작 !! 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66)


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
# print(x_train.shape) #(398, 30)
# print(x_test.shape)  #(171, 30)
# print(y_test.shape)  #(171,)
# print(y_train.shape)  #(398,)
# print(y_test)


x_train = x_train.reshape(398, 30, 1) 
x_test = x_test.reshape(171, 30, 1) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(30,1)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.1)
model.compile(loss='mse',optimizer=optimizer,
                    metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
start_time = time.time()
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
#es는 조건이맞으면 끝나고, reduce_lr도 조건 맞으면 끝나지만, factor=0.5 로 해놓으면 감소가 없으면 러닝 레이트가 0.5 만큼 줄어든다.

hist = model.fit(x_train, y_train, epochs=300, verbose=1,validation_split=0.2,
batch_size=1, callbacks=[es, reduce_lr])


end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print('acc : ', loss[1])

#acc :  acc :  0.9298245906829834
#acc :  0.9766082167625427