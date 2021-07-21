import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

print(datasets.feature_names)

print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))
#2 모델구성

#3. 컴파일 훈련

#4. 평가, 예측 
# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle=True, random_state=66)

# model 구성 
model = Sequential()
model.add(Dense(1, input_dim=10))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(1))

#model.save('./_save/keras46_1_save_model_1.h5')
model.save_weights('./_save/keras46_1_save_weights_1.h5')


model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100, verbose= 1, batch_size=3, validation_split=0.2, shuffle=True, callbacks=[es])
end_time = time.time() - start_time

#model.save('./_save/keras46_1_save_model_2.h5')
model.save_weights('./_save/keras46_1_save_weights_2.h5')
#save_weights 에는 모델이 저장되지 않는다 



loss = model.evaluate(x_test, y_test)
print("경과시간 :", end_time)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 : ' , r2)


