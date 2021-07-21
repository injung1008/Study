#실습
# 1_100까지의 데이터를
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.metrics import r2_score

a = np.array(range(1,101))
x_predict = np.array(range(96, 105))
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)

# print(dataset)

x = dataset[:,:5]
y = dataset[:, 5]
x = x.reshape(95,5,1)

# x_predict = x_predict[-4:]
# print(x_predict)
# x_predict = x_predict.reshape(1,4,1)
# print(x_predict.shape)

def split_y(x,y,x_predict):

    for i in range(2):
        x_predict = x_predict[-5:]
        print('f')
        x_predict = x_predict.reshape(1,5,1)
        model = Sequential()
        #model.add(LSTM(units=100, activation='relu', input_shape=(5,1)))
        model.add(Conv1D(64, kernel_size=2, input_shape=(5,1)))
        print('d')
        model.add(LSTM(64, return_sequences=True)) #들어오는 쉐이프만 잘 맞춰주면 된다
        model.add(Conv1D(64,2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Dense(4))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(x,y, epochs=20, batch_size=1)
        results = model.predict(x_predict)
        #x_predict = x_predict.reshape(1,4)
        print(results)
        x_predict = np.append(x_predict,results)

    return results

split_y(x, y, x_predict)





