from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



#1. 데이터
x = np.array([range(10)]) 
x = np.transpose(x)


y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10,9,8,7,6,5,4,3,2,1],
            [2,4,6,8,10,12,14,16,18,20]]) # 3행 10열 
y = np.transpose(y) # (10, 3) x와 key : value를 맞춰줘야 하기 때문에 표를 맞춰 줘야 한다 

print(x.shape)
print(y.shape)


#2. 모델 만들기
model = Sequential()
model.add(Dense(2, input_dim=1)) # x의 열 
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(4))

#3. 컴파일 훈련시키기
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=10000, batch_size=1)

#4. loss값 뽑기 (평가 예측 )
loss = model.evaluate(x,y)


print('loss : ', loss)

result = model.predict(x)
print('예측값 : ', result)
