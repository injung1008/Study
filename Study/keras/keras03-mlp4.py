from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
xcol1 = x[0] # (3행 10열)
xcol2 = x[1]
xcol3 = x[2]
x = np.transpose(x)# 10행 3열

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10,9,8,7,6,5,4,3,2,1]]) # 3행 10열 
ycol1 = y[0]
ycol2 = y[1]
ycol3 = y[2]
y = np.transpose(y) # (10, 3) x와 key : value를 맞춰줘야 하기 때문에 표를 맞춰 줘야 한다 
print(x.shape)
print(y.shape)


# #2. 모델 만들기
# model = Sequential()
# model.add(Dense(3, input_dim=3)) # x의 열 
# #Dense(3 = y의 갯수 

# #3. 컴파일 훈련시키기
# model.compile(loss='mse', optimizer='adam')
# model.fit(x,y, epochs=10, batch_size=1)

# #4. loss값 뽑기 (평가 예측 )
# loss = model.evaluate(x,y)


# print('loss : ', loss)

#result = model.predict([[0,21,201]])
# print('예측값 : ', result)

model = Sequential()
model.add(Dense(3, input_dim=1))
model.compile(loss='mse', optimizer='adam')
model.fit(xcol1,ycol1, epochs=100, batch_size=1)
model.fit(xcol1,ycol2, epochs=100, batch_size=1)
model.fit(xcol1,ycol3, epochs=100, batch_size=1)
model.fit(xcol2,ycol1, epochs=100, batch_size=1)
model.fit(xcol2,ycol2, epochs=100, batch_size=1)
model.fit(xcol2,ycol3, epochs=100, batch_size=1)
model.fit(xcol3,ycol1, epochs=100, batch_size=1)
model.fit(xcol3,ycol2, epochs=100, batch_size=1)
model.fit(xcol3,ycol3, epochs=100, batch_size=1)
y_result = model.predict(xcol1) # x데이터 전체의 데이터 예측값이 나온다  (x,y)가 나오면서 이어지며 예측값을 이 나오게 된다 


plt.scatter(xcol1,ycol1,ycol2,ycol3)
plt.plot(xcol1,y_result, color='red')
plt.show()

y_result = model.predict(xcol2) # x데이터 전체의 데이터 예측값이 나온다  (x,y)가 나오면서 이어지며 예측값을 이 나오게 된다 


plt.scatter(xcol2,ycol1,ycol2,ycol3)
plt.plot(xcol2,y_result, color='red')
plt.show()

y_result = model.predict(xcol3) # x데이터 전체의 데이터 예측값이 나온다  (x,y)가 나오면서 이어지며 예측값을 이 나오게 된다 


plt.scatter(xcol3,ycol1,ycol2,ycol3)
plt.plot(xcol3,y_result, color='red')
plt.show()


