
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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(1, input_dim=30))
model.add(Dense(100, activation='relu'))
model.add(Dense(123, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #0-1사이의 값이 출력이 된다 

model.compile(loss='binary_crossentropy', optimizer='adam'
                        , metrics=['mse','accuracy']) 
#mse를 사용하면 선을 긋고 비교를 하기때문에 이진 분류에서는 loss='binary_crossentropy 사용

from tensorflow.keras.callbacks import EarlyStopping
#monitor를 기준으로 멈추겠다는 얘기 
# patience = loss값이 갱신되지 않는것을 5번까지는 참겠다, patience는 훈련수를 넘지 않는다
#mode = 'min' 로스 최저값이 5번을 갱신하면 멈추겠다 
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=50, verbose=1, 
            batch_size=1, validation_split=0.2 ,callbacks=[es])
# print(hist.history['loss'])
# print(hist.history['val_loss'])

print("=======평가예측==========")
loss = model.evaluate(x_test, y_test)
#이제 여기서 나오는 loss는 binary crossentropy이다 
#evaluate는 loss와 metrix도 리스트 형태로 같이 반환해준다 
print('loss : ', loss )
print('loss : ', loss[0] )
print('metrix : ', loss[1] )


print(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
print(y_predict)
