from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time


#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10,9,8,7,6,5,4,3,2,1]]) # 3행 10열 
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)

x = np.transpose(x) #10행 3열 (짝을 지어 줘야 하기때문에 y와 )
print(y.shape)

#2. 모델 만들기
model = Sequential()
model.add(Dense(1, input_dim=3)) # x의 열 

#3. 컴파일 훈련시키기
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x,y, epochs=10, batch_size=2, verbose=1)
end = time.time() - start 
print("걸린시간 : ", end)



#4. loss값 뽑기 (평가 예측 )
loss = model.evaluate(x,y)


print('loss : ', loss)

result = model.predict([[10,1.3,1]])
print('예측값 : ', result)
print(x.shape)

#1. mae지표 찾기
#2. rmse착지 