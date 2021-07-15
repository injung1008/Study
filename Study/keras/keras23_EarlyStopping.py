#epochs를 많이 돌리면 과적합이 발생하게 되는데 조금 돌리면 제대로 훈련을 
#하지못해 예측을 할 수 없다 이때 훈련을 많이 돌리고 훈련을 중간에 하지 못하게
#멈출 수 있는데 이것을 early stopping이라고 하며 성능이 어느순간 증가 되지 않는 그 시점에서 멈추게 한 다
# loss의 최저점, 


from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target

#print(np.min(x), np.max(x)) # 총 데이터에서 최솟값 =0.0 최댓값 = 711.0

#데이터 전처리 (최댓값으로 x를나눠준다 - min-max- scaler방법)
# x = x/711.
# x = x/np.max(x)
#x = (x - np.min(x))/ (np.max(x) - np.min(x)) #전체 데이터 정규화
x_train, x_test, y_train, y_test = train_test_split(x, 
                        y, train_size = 0.7, shuffle=True, random_state=66)
from sklearn.preprocessing import StandardScaler #표준정규분포

#표준정규분포화로 바꿔줌 
scaler = StandardScaler()
#fit = (전처리) 무언가를 훈련,실행 시키다의 개념 공식을 만들어준다 
scaler.fit(x_train)
# 실행 시킨 결과를 변환시 키는 과정 (배출의 과정)
x_train = scaler.transform(x_train)
print(x_train)
#스케일릴ㅇ 비율에 맞춰서 트랜스폼 된것이다 
x_test = scaler.transform(x_test)


print(x.shape) #(506,13)
print(x_train.shape) #(354,13)
print(y.shape)

# model 구성 
model = Sequential()
model.add(Dense(1, input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(123, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
#monitor를 기준으로 멈추겠다는 얘기 
# patience = loss값이 갱신되지 않는것을 5번까지는 참겠다, patience는 훈련수를 넘지 않는다
#mode = 'min' 로스 최저값이 5번을 갱신하면 멈추겠다 
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=50, verbose=1, 
            batch_size=1, validation_split=0.2 ,callbacks=[es])
print(hist.history['loss'])
print(hist.history['val_loss'])

print("=======평가예측==========")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)

#"C:\Windows\Fonts/gulim.ttc"

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:\Windows\Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(hist.history['loss'])  #x: eposch / y : hist.history['loss]
plt.plot(hist.history['val_loss'])

plt.title("오류, 검정오류") #타이틀 명 
plt.xlabel('epochs')  #x축 이름 
plt.ylabel('loss, val_loss') #축 이름
plt.legend(['train loss', 'val loss']) #범례
plt.show()


