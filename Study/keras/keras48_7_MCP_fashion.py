import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) - 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# #scaler = MinMaxScaler()
scaler = StandardScaler()
# #scaler = MaxAbsScaler()
# #scaler = RobustScaler()
# scaler = QuantileTransformer()
# #scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(60000, 784,1) 
x_test = x_test.reshape(10000, 784, 1) 


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
#print(y_test.shape)

#2. 모델링
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

# model = Sequential()
# model.add(Conv1D(64, 2,input_shape=(784,1)))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(10, activation='softmax')) # 왜 시그모이드를 사용할까? 

# model.compile(loss='categorical_crossentropy',optimizer='adam',
#                     metrics=['accuracy'])


# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
#                     filepath='./_save/ModelCheckPoint/keras47_MCP_fashion.hdf5')

# import time
# start_time = time.time()
# model.fit(x_train, y_train, epochs=30, verbose= 1, batch_size=3,
#              validation_split=0.2, callbacks=[es,cp])

# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras47_fashion_save.h5')
#model = load_model('./_save/ModelCheckPoint/keras47_fashion_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_fashion.hdf5')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )


#metrix :  0.8108000159263611
'''
nomal
loss :  [0.5505011081695557, 0.8158000111579895]
metrix :  0.8158000111579895

save 
loss :  [0.5505011081695557, 0.8158000111579895]
metrix :  0.8158000111579895

MCP
loss :  [0.5416776537895203, 0.8183000087738037]
metrix :  0.8183000087738037

'''