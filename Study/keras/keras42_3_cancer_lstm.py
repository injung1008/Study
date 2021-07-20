
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(569, 30) (569,)
# print(y[:20])
# #[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y)) #[0 1] 중복되는게 없는 유니크한 값

#실습 : 모델 시작 !! 

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
# print(x_train.shape) #(398, 30)
# print(x_test.shape)  #(171, 30)
# print(y_test.shape)  #(171,)
# print(y_train.shape)  #(398,)
# print(y_test)


x_train = x_train.reshape(398, 30, 1) 
x_test = x_test.reshape(171, 30, 1) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(30,1)))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam'
                         , metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=10, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print('acc : ', loss[1])

#acc :  acc :  0.9298245906829834