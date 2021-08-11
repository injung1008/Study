# 다중분류 ( y가 0,1,2로 이루어져 있을때)
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #LogisticRegression은 로지스틱회귀분석 :  분류모델 
from sklearn.tree import DecisionTreeClassifier #의사결정 나무 = 분류모델과 회귀모델 이 있다
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트는 앙상블 모델이고 앙상블에는 배깅과 부스트가 있다 
import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 

datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150,4) (150,)
# print(y) # y = 0,1,2




from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66)


kfold = KFold(n_splits=5, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001,0.0001]}
]

#!2. model 구성 
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)

# Fitting 5 folds for each of 18 candidates, totalling 90 fits -> 그리드 서치
# Fitting 5 folds for each of 10 candidates, totalling 50 fits -> 랜덤 

#3.훈련
model.fit(x_train,y_train)

#4. 평가, 예측 
print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 

print("best_score_ : ", model.best_score_)

print("model.score :", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_pred))



