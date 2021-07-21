
from tensorflow.keras.layers import Dense, LSTM, Conv1D,Flatten
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target


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



x_train = x_train.reshape(398, 30, 1) 
x_test = x_test.reshape(171, 30, 1) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# model = Sequential()
# model.add(Conv1D(64, 2, input_shape=(30,1)))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam'
#                          , metrics=['accuracy']) 

#model = load_model('./_save/ModelCheckPoint/keras47_cancer_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_cancer.hdf5')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/ModelCheckPoint/keras47_MCP_cancer.hdf5')


model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=10, validation_split=0.2)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=30, verbose= 1, batch_size=3,
             validation_split=0.2, callbacks=[es,cp])

end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras47_cancer_save.h5')

loss = model.evaluate(x_test, y_test)
print('acc : ', loss[1])


'''
기본 
acc :  0.9766082167625427

save_point
acc :  0.9766082167625427

MCP
acc :  0.9064327478408813

'''
