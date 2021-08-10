
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size = 0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#실습, 모델 구성하고 완료하시오
#회귀 데이터를 classfier로 만들었을 경우에 에러 확인 !! 


from sklearn.svm import LinearSVC, SVC #가능한지 확인 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #LogisticRegression은 로지스틱회귀분석 :  분류모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor #의사결정 나무 = 분류모델과 회귀모델 이 있다
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# model = LinearSVC() #LinearSVC 모델을 만든다 / 컴파일도 포함되어 있다 
#회귀데이터라서 불가능
# model = SVC()
#회귀데이터라서 불가능
# model = KNeighborsClassifier()
##회귀데이터라서 불가능

# model = KNeighborsRegressor()
#result :  0.8255996350265673
# model = LogisticRegression()
##회귀데이터라서 불가능
# model = LinearRegression()
#result :  0.8044888426543628
# model = DecisionTreeClassifier()
#회귀데이터라서 불가능
# model = DecisionTreeRegressor()
#result :  0.7230533880152936
# model = RandomForestClassifier()
#회귀데이터라서 불가능
model = RandomForestRegressor()
#result :  0.8882353549048055
#! 랜덤 스테이트를 바꾸면 훈렌데이터셋도 바뀌기때문에 이에 따라 acc또한 바뀌게 된다 


#3. 훈련
model.fit(x_train, y_train) 
# ValueError: y should be a 1d array, got an array of shape (105, 3) instead.
#! -> 사이킥런의 대부분은 원핫인코딩 할필요가 없다 svc에서 원핫인코딩까지 해주겠다

#사이킥런은 평가하는것이 기존 evaluate -> score이다  
results = model.score(x_test, y_test)
print('result : ',results)

from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)









