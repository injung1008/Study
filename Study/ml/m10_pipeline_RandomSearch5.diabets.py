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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #랜덤포레스트는 앙상블 모델이고 앙상블에는 배깅과 부스트가 있다 
import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 
from sklearn.datasets import load_diabetes
datasets = load_diabetes()

from sklearn.pipeline import make_pipeline, Pipeline #popeline 임포트 

x = datasets.data
y = datasets.target


from sklearn.model_selection import RandomizedSearchCV,train_test_split, KFold, cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66)


kfold = KFold(n_splits=5, shuffle=True, random_state=66)

#RandomForestClassifier()) 의 파라미터 
parameters = [
    {'randomforestregressor__n_estimators' : [100,200]},
    {'randomforestregressor__max_depth' : [6, 8, 10, 12]}
]

#!2. model 구성 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())

model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)
# model 넣을 자리에 pipe정의해준거 넣어주기 랜덤포레스트 모델에 맞는 파라미터가 들어와야하는데 파이프가 들어왔으니
# 파이프에 맞게 파라미터를 수정해줘야한다. 
# 파이프 라인에서 쓰는 모델(randomforestclassifier)을 파라미터 앞에 붙여주면 된다  'n_estimators'  -> 'randomforestclassifier__n_estimators'

#3.훈련
model.fit(x_train,y_train)

#4. 평가, 예측 
print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 
#best_estimator_ 가장 좋은 평가가 무엇인가? 
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("best_score_ : ", model.best_score_)
#가장 좋은 값을 출력해준다 best_score_ :  0.9800000000000001

# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수 :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestregressor',
#                  RandomForestRegressor(n_estimators=200))])
# best_score_ :  0.43154191358581856
# PS D:\Study>