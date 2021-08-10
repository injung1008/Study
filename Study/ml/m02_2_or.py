from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. 데이터 (or gate)
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,1]
# and gate  둘중에 하나라도 0 이면 0 이고 둘다 1일경우 1이다 
# or gate  둘중에 하나라도 1이면 1이다 둘다 0일 경우만 0 
# xor gate 둘이 같으면 0,0 = 0  / 1,1, = 0  이고 둘이 다를경우 0,1 =1 이다 


#2. model 
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측 
y_predict = model.predict(x_data)
print(x_data, "의 예측 결과 : ", y_predict)
#[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과 :  [0 1 1 1]

results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score :", acc)