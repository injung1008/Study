import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_csv('./_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

datasets = df.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]
# print(x.shape)
# print(y.shape)
# print(np.unique(y)) #7개 [3 4 5 6 7 8 9]


x_train, x_test, y_train, y_test =train_test_split(x,y,
    train_size = 0.7, shuffle = True, random_state=2)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape) #(105, 4)
# print(x_test.shape)  #(45, 4)
# print(y_test.shape)  #(45, 3)
# print(y_train.shape)  #(105, 3)
# print(y_test)



#!model 구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #LogisticRegression은 로지스틱회귀분석 :  분류모델 
from sklearn.tree import DecisionTreeClassifier #의사결정 나무 = 분류모델과 회귀모델 이 있다
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트는 앙상블 모델이고 앙상블에는 배깅과 부스트가 있다 
#의사결정 나무를 합쳐서 랜덤포레스트를 만들기때문에 의사결정나무의 확장판이라고 생각하면된다. 

model = LinearSVC() #LinearSVC 모델을 만든다 / 컴파일도 포함되어 있다 
#auuracy_score :  0.5319727891156463
# model = SVC()
#auuracy_score :  0.5340136054421769
# model = KNeighborsClassifier()
#auuracy_score :  0.5435374149659864
# model = LogisticRegression()
#auuracy_score :  0.5231292517006803
# model = DecisionTreeClassifier()
#auuracy_score :  0.5782312925170068
# model = RandomForestClassifier()
# auuracy_score :  0.6489795918367347

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
# print(y_predict)
acc = accuracy_score(y_test, y_predict)
print("auuracy_score : ", acc)


# print('==========예측=============')
# print(y_test[:5])
# y_predict2 = model.predict(x_test[:5])
# print(y_predict2) #[0 1 1 0 2]




