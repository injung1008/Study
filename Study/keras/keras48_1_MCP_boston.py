
#보스톤 -> lstm 분석 

from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential, load_model
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



# model = Sequential()
# model.add(Conv1D(64,kernel_size=2,input_shape=(13,1)))
# model.add(Flatten())
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(12))
# model.add(Dense(1)) 

# model.compile(loss='mse',optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
#                     filepath='./_save/ModelCheckPoint/keras47_MCP_boston.hdf5')

# import time
# start_time = time.time()
# model.fit(x_train, y_train, epochs=30, verbose= 1, batch_size=3,
#              validation_split=0.2, callbacks=[es,cp])

# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras47_boston_save.h5')
#model = load_model('./_save/ModelCheckPoint/keras47_boston_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_boston.hdf5')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
#print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

'''
일반
loss :  16.88153648376465
r2 :  0.7956653557579908

save_model
loss :  16.88153648376465
r2 :  0.7956653557579908

MCP
loss :  17.037504196166992
r2 :  0.7937774969569851


'''


