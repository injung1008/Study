from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target
 
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)


# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=66)

# model 구성 

input1 = Input(shape=(13,)) #인풋레이어 구성
dense1 = Dense(2)(input1) #히든 레이어 구성 
dense2 = Dense(40)(dense1)
dense3 = Dense(160)(dense2)
dense4 = Dense(220)(dense3)
dense5 = Dense(168)(dense4)
output1 = Dense(1)(dense5) 

model = Model(inputs=input1, outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=5)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)




