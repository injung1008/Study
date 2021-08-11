# 실습 
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
# x_train, x_test, y_train, y_test = train_test_split(x,y,
#         train_size = 0.7, shuffle = True, random_state=66)


#GridSearchCV - 모델, 러닝레이트, 크로스발리데이션 까지 한꺼번에 랩으로 싸겠다. 그리고 이 자체를 하나의 모델로 본다 
#이 모델은 기존 모델과 우리가 넣고싶은 하이퍼 파라미터의모임을 딕셔너리 형태로 구성해주고 크로스 발리데이션 까지명시해주면 모델의 형태가 된다
#크로스 발리데이션만큼 돌아간다 cv를 5번 주면 모델이 5번 돌아가고 노드가 3개면 15번 돌아가고 러닝레이트가 3이면 45번 돌아간다 


kfold = KFold(n_splits=5, shuffle=True, random_state=66)
#n_splits=5 전체 데이터를 5등분해서 5번 반복해준다 = 20% test  (등분한 수만큼 반복을 해준다) 
#cross_val_score = 교차검증 방법으로 kfold와비슷 

parameters = [
    {'n_estimators' : [100,200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

#!2. model 구성 
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
#파라미터와 cv를 곱한것만큼 돌아간다 SVC에는 여러가지 파라미터가 존재한다 
#gridSearch에서는 fit을 지원한다 
# model = SVC()

#3.훈련
model.fit(x,y)
#저 파라미터중 어떤 파라미터가 가장 좋은값을 내는지 확인하는게 중요하다 
#이후 가장 좋은 파라미터를 가지고 다시 훈련을 시키면 된다 

#4. 평가, 예측 
print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 
#best_estimator_ 가장 좋은 평가가 무엇인가? 
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("best_score_ : ", model.best_score_)
#가장 좋은 값을 출력해준다 best_score_ :  0.9800000000000001

#! 그리드
# 최적의 매개변수 :  RandomForestClassifier()
# best_score_ :  0.9666666666666666

#! 랜덤
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=10)
# best_score_ :  0.9666666666666666