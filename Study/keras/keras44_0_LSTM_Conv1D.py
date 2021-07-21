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


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.7, shuffle=True, random_state=9)
x_test = x_test.reshape(29,4,1)
x_train = x_train.reshape(67,4,1)
#y_test = x_test.reshape(29,4,1)
y_train = x_train.reshape(67,4,1)

print(y_train.shape)


x_predict = x_predict[-4:]
print(x_predict)
x_predict = x_predict.reshape(1,4,1)
print(x_predict.shape)

def split_y(x_test,x_train,y_test,y_train,x_predict):

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
        model.fit(x_train, y_train, epochs=2, batch_size=1)
        results = model.predict(x_predict)
        print('results 1 :', results)
        x_predict = np.append(x_predict,results)
        result2 = model.predict(x_test)
        print('result2 : ',result2)
        r2 = r2_score(y_test, result2)
        print('r2 : ', r2)     
        
    return result2, results

split_y(x_test,x_train,y_test,y_train,x_predict)











