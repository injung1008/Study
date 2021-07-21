#실습
 #106 예측하기 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score

a = np.array(range(1,101))
x_predict = np.array(range(96, 106))
size = 6
#

def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
#print(dataset.shape) #(95, 6)

x_train = dataset[:,:5].reshape(95,5,1)
y_train = dataset[:, 5]
#print(x_train.shape) #(95, 5)
#print(x_predict) #[ 96  97  98  99 100 101 102 103 104 105] 

test_set = split_x(x_predict,size)

x_test = test_set[:,:5].reshape(5,5,1)
#print(x_test.shape) #(5, 5)

y_test = test_set[:,5]
#print(y_test)

model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(5,1)))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1)
results = model.predict(x_test)
print(x_test)
print(y_test)
print(results)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, results)
print('r2 : ', r2)    

'''
[[100.943436]
 [101.942314]
 [102.94121 ]
 [103.94014 ]
 [104.93909 ]]
r2 :  0.9982722655113321
'''