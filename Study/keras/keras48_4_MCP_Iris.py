# 다중분류 ( y가 0,1,2로 이루어져 있을때)
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import numpy as np


datasets = load_iris()

x = datasets.data
y = datasets.target



from tensorflow.keras.utils import to_categorical #one hot 라이브러리
y = to_categorical(y) #위에서 벡터로 바꿔주는 과정을 얘가 처리해줌 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66,stratify=y)



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

x_train = x_train.reshape(105, 4, 1) 
x_test = x_test.reshape(45, 4, 1) 



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# model = Sequential()
# model.add(Conv1D(64, 2, input_shape=(4,1)))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(3, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam'
#                          , metrics=['accuracy']) 

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# import time
# start_time = time.time()

# es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras47_MCP_iris.hdf5')


# model.fit(x_train, y_train, epochs=50, verbose=1, 
#             batch_size=1, validation_split=0.2 ,callbacks=[es,cp])

# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras47_iris_save.h5')

#model = load_model('./_save/ModelCheckPoint/keras47_iris_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_iris.hdf5')


print("=======평가예측==========")
loss = model.evaluate(x_test, y_test)

print('loss : ', loss )
print('metrix : ', loss[1] )





'''
일반
loss :  [0.1032024621963501, 0.9555555582046509]      
metrix :  0.9555555582046509
save
loss :  [0.1032024621963501, 0.9555555582046509]      
metrix :  0.9555555582046509

MCP
loss :  [0.08631491661071777, 1.0]
metrix :  1.0

'''