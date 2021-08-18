#실습 
#데이터 diabets
#1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성 
# 최적의 R2값과 피쳐임포턴스 구할것 

#2 위 스레드값으로 SelectFromModel돌려서 최적의 피쳐갯수 구할것 

#3. 위 피쳐 갯수로 피쳐 갯수를 조정한뒤 그걸로 다시 랜덤서치 그리드서치해서 
#최적의 R2 구할것 


from xgboost import XGBRegressor 
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
import numpy as np 
from sklearn.metrics import r2_score, accuracy_score 

# datasets = load_boston() 
# x = datasets.data 
# y = datasets.target    

x,y = load_diabetes(return_X_y=True) #x,y 분리해서 출력해줌 


# from sklearn.decomposition import PCA

# pca = PCA(n_components=10) #총 10개 칼럼을 7개로 압축하겠다 
# x = pca.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=9
        )
#2. 모델 


parameters = [
    {'n_estimators' : [100,200,300]},
    {'learning_rate' : [0.1,0.01,0.05,0.5]},
    {'n_jobs' : [-1,1, 2, 4]}
]


# model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1)

# model = XGBRegressor(booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,     
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=6,
#              min_child_weight=1,monotone_constraints='()',     
#              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,     
#              tree_method='exact', validate_parameters=1, verbosity=None)

model = GridSearchCV(estimator=XGBRegressor(), param_grid=parameters, 
  verbose=1)

# model = GridSearchCV(XGBRegressor(), parameters)
#!2. model 구성 

#3. 훈련 
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
print("model score : ", score)  #기본으로 r2 score를 제공한다 





print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 




# # 컬럼 하나씩 삭제하고 값 확인하기 
# from sklearn.feature_selection import SelectFromModel 
# thresholds= np.sort(model.feature_importances_)
# print(thresholds) 

# for thres in thresholds:
#     selection = SelectFromModel(model, threshold=thres, prefit=True)
#     #thresholds= > 예시 0.00363372 이상의 칼럼들만 훈련시킴 그럼 첫번째 칼럼 사라지고 12개 진행 
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape)

#     selection_model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1)
#     selection_model.fit(select_x_train, y_train)

#     y_pred = selection_model.predict(select_x_test)

#     score = r2_score(y_test, y_pred)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thres, 
#     select_x_train.shape[1], score*100
#     ))
