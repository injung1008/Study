from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#1. 데이터 

#훈련 50%/ 검증 30% 테스트 20% 

# #훈련용
x = np.array(range(100))
y = np.array(range(1,101))

for i in range(1,10):
    random_state = i
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=i)

    model = Sequential()
    model.add(Dense(5, input_dim=1)) 
    model.add(Dense(1))


    #3. 컴파일 훈련 
    model.compile(loss='mse', optimizer='adam')

    model.fit(x_train, y_train, epochs=1, batch_size=1) 
    # -> 이 과정에서 가중치가 생성이 된다 

    #4. 평가, 예측 
    loss = model.evaluate(x_test, y_test)
    print('loss : ', loss)

    y_predict = model.predict(x_test) 
    print(y_predict)

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_predict)
    print(r2)

    def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))

    rmse = RMSE(y_test, y_predict)
    print("rmse 스코어 : ", rmse)



