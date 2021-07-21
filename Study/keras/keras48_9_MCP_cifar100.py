import numpy as np
import matplotlib.pyplot as plt

#1. 데이터 
from tensorflow.keras.datasets import cifar100
(x_train, y_train),(x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# #전처리 

print(np.unique(y_train)) #[0- 100]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() 
y_test = encoder.transform(y_test).toarray() 
#print(y_test.shape)

#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000*3, 32*32)
x_test = x_test.reshape(10000*3, 32*32)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
# scaler = QuantileTransformer()
# #scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(50000, 1024, 3) 
x_test = x_test.reshape(10000, 1024, 3) 

#2. 모델링
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM,Conv1D,Flatten
# model = Sequential()
# model.add(Conv1D(512,2, input_shape=(1024,3)))
# model.add(Flatten())
# model.add(Dense(264))
# model.add(Dense(182))
# model.add(Dense(100))
# model.add(Dense(100, activation='softmax')) 

# model.compile(loss='categorical_crossentropy',optimizer='adam',
#                     metrics=['accuracy'])


# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
#                     filepath='./_save/ModelCheckPoint/keras47_MCP_cifar100.hdf5')

# import time
# start_time = time.time()
# model.fit(x_train, y_train, epochs=50, verbose= 1, batch_size=300,
#              validation_split=0.2, callbacks=[es,cp])

# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras47_cifar100_save.h5')
#model = load_model('./_save/ModelCheckPoint/keras47_cifar100_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_cifar100.hdf5')


loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )

#accuracy: 0.2709
'''
loss :  [3.9171793460845947, 0.15309999883174896]
metrix :  0.15309999883174896

loss :  [3.9171793460845947, 0.15309999883174896]  
metrix :  0.15309999883174896

loss :  [3.864924907684326, 0.14900000393390656]   
metrix :  0.14900000393390656

'''