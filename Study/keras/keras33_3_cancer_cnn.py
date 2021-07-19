
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,)
print(y[:20])
#[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
print(np.unique(y)) #[0 1] 중복되는게 없는 유니크한 값

#실습 : 모델 시작 !! 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

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
# print(x_train.shape) #(398, 30)
# print(x_test.shape)  #(171, 30)
# print(y_test.shape)  #(171,)
# print(y_train.shape)  #(398,)
# print(y_test)


x_train = x_train.reshape(398, 10,3, 1) 
x_test = x_test.reshape(171, 10,3, 1) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(100, kernel_size=(2,1), padding='same', input_shape=(10,3,1)))
model.add(Conv2D(100, (2,1)))
model.add(Flatten())                                              
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam'
                         , metrics=['accuracy']) 

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=10, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print('acc : ', loss)
