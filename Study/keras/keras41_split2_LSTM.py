#실습
# 1_100까지의 데이터를
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score

a = np.array(range(1,101))
x_predict = np.array(range(96, 105))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)

# print(dataset)

x = dataset[:,:4]
y = dataset[:, 4]
x = x.reshape(96,4,1)

# x_predict = x_predict[-4:]
# print(x_predict)
# x_predict = x_predict.reshape(1,4,1)
# print(x_predict.shape)

def split_y(x,y,x_predict):

    for i in range(2):
        x_predict = x_predict[-4:]
        print('f')
        x_predict = x_predict.reshape(1,4,1)
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', input_shape=(4,1)))
        model.add(Dense(8))
        model.add(Dense(4))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(x,y, epochs=2, batch_size=1)
        results = model.predict(x_predict)
        #x_predict = x_predict.reshape(1,4)
        print(results.shape) #(1, 1)
        print(x_predict.shape) #(1, 4, 1)
        #넘파이에 추가할때는 np.append(넘파이, 추가할값) 
        # -> shape가 데이터 배열로만으로 바뀌게된다
        x_predict = np.append(x_predict,results)
        print(x_predict.shape) #(5,)

        
    return results

split_y(x, y, x_predict)




