from xgboost import XGBRegressor 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score, accuracy_score 

# datasets = load_boston() 
# x = datasets.data 
# y = datasets.target    

x,y = load_boston(return_X_y=True) #x,y 분리해서 출력해줌 
print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=66
        )
#2. 모델 

model = XGBRegressor(n_jobs=8)

#3. 훈련 
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
print("model score : ", score)  #기본으로 r2 score를 제공한다 

# model score :  0.9221188601856797PS

# aaa= model.feature_importances_
# print(aaa) # 기존은 칼럼의 순서대로 정렬이 되어있다. 
#[0.01447935 0.00363372 0.01479119 0.00134153 0.06949984
#  0.30128643 0.01220458 0.0518254  0.0175432 
# 0.03041655 0.04246345 0.01203115 0.42848358]

# aaa= np.sort(model.feature_importances_)
# print(aaa) 
#0.00134153 0.00363372 0.01203115 0.01220458 0.01447935
#  0.01479119 0.0175432  0.03041655 0.04246345 0.0518254 
#  0.06949984 0.30128643 0.42848358]
#앞에꺼보다 한개씩 한개씩 삭제를 하면서 모든 결과치가 나오게 한다 

#과적합 방지 = 훈련데이터 늘린다, 레이어의 노드 줄이기(Dropout)-딥러닝 개념
# ,lasso규제(Nomalization,regulrization)

#컬럼 하나씩 삭제하고 값 확인하기 
from sklearn.feature_selection import SelectFromModel 
thresholds= np.sort(model.feature_importances_)
print(thresholds) 

for thres in thresholds:
    selection = SelectFromModel(model, threshold=thres, prefit=True)
    #thresholds= > 예시 0.00363372 이상의 칼럼들만 훈련시킴 그럼 첫번째 칼럼 사라지고 12개 진행 
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thres, 
    select_x_train.shape[1], score*100
    ))

# model score :  0.9221188601856797 #기존 하나도 안지웠을떄 score
#<featue importance>
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935
#  0.01479119 0.0175432  0.03041655 0.04246345 0.0518254 
#  0.06949984 0.30128643 0.42848358]
# (404, 13)
# Thresh=0.001, n=13, R2: 92.21%
# (404, 12)
# Thresh=0.004, n=12, R2: 92.16%
# (404, 11)
# Thresh=0.012, n=11, R2: 92.03%
# (404, 10)
# Thresh=0.012, n=10, R2: 92.19%
# (404, 9)
# Thresh=0.014, n=9, R2: 93.08%  #4개 삭제할때 성능  최고치 
# (404, 8)
# Thresh=0.015, n=8, R2: 92.37%
# (404, 7)
# Thresh=0.018, n=7, R2: 91.48%
# (404, 6)
# Thresh=0.030, n=6, R2: 92.71%
# (404, 5)
# Thresh=0.042, n=5, R2: 91.74%
# (404, 4)
# Thresh=0.052, n=4, R2: 92.11%
# (404, 3)
# Thresh=0.069, n=3, R2: 92.52%
# (404, 2)
# Thresh=0.301, n=2, R2: 69.41%
# (404, 1)
# Thresh=0.428, n=1, R2: 44.98%


