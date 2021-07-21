import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential, load_model
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




# model = Sequential()
# model.add(Conv1D(10,2, input_shape=(11,1)))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(7, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam'
#                          , metrics=['accuracy']) 


# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# cp = ModelCheckpoint(monitor = 'val_loss', save_best_only=True, mode='auto',
#                     filepath='./_save/ModelCheckPoint/keras47_MCP_wine.hdf5')


# model.fit(x_train, y_train, epochs=10, verbose=1,
# batch_size=100, validation_split=0.02, callbacks=[es,cp])

# model.save('./_save/ModelCheckPoint/keras47_wine_save.h5')
#model = load_model('./_save/ModelCheckPoint/keras47_wine_save.h5')

model = load_model('./_save/ModelCheckPoint/keras47_MCP_wine.hdf5')


loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )

'''
일반 
 loss: 1.2455 - accuracy: 0.4748
loss :  [1.2034642696380615, 0.49251699447631836] 
metrix :  0.49251699447631836

save
loss :  [1.2034642696380615, 0.49251699447631836]
metrix :  0.49251699447631836 

loss :  [1.2034642696380615, 0.49251699447631836]
metrix :  0.49251699447631836        
'''