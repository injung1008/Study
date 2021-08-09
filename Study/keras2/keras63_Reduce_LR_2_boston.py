
#보스톤 -> lstm 분석 

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size = 0.7, shuffle=True, random_state=66)
# print(x.shape) #(506,13)
# print(x_train.shape) #(354,13)
# print(y.shape)
#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape) #(354, 13)
# print(x_test.shape)  #(152, 13)
# print(y_test.shape)  #(152,)
# print(y_train.shape)  #(354,)


x_train = x_train.reshape(354, 13,1) 
x_test = x_test.reshape(152, 13,1) 

print(y_test.shape) #(152,)

# # model 구성 



model = Sequential()
model.add(Conv1D(64,kernel_size=2,input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(12))
model.add(Dense(1)) 


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
batch_size=10, callbacks=[es, reduce_lr])


end_time = time.time() - start_time


loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

#r2 : r2 :  0.789056554938404
#r2 :  0.8000906834001942
#r2 :  0.8161368222220343

# -러닝 레이트 
#r2 r2 :  0.787588830791354


