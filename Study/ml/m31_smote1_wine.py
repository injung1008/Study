#pip install smote
from imblearn.over_sampling import SMOTE  
from sklearn.datasets import load_wine 
import pandas as pd 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(178, 13) (178,)

print(pd.Series(y).value_counts())
# 1    71     
# 0    59     
# 2    48     
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

#끝에서 30개를 제외한 178 -> 148 개만 뽑음 
x_new = x[:-30]
y_new = y[:-30]

print(x_new.shape, y_new.shape) #(178, 13) (178,) ->(148, 13) (148,)
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train,y_test = train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=66,stratify=y_new
)
#랜덤스테이트를 바꿀때마다 달라진다 라벨 갯수가 달라진다
# stratify=y_new 로 해놓으면 각 y의 라벨의 비율만큼 동일하게 나온다 

print(pd.Series(y_train).value_counts())
                 #    미래 증폭 (제일 큰 라벨 값에 맞춰서 증폭하기)
# 1    71 -> | 1    53   | -> 53
# 0    59 -> | 0    44   | -> 53
# 2    18 -> | 2    14   | -> 53 
                 #     total 159개     
#테스트는 증폭을 해야할까? -> 증폭할 필요가 없다 

#데이터 증폭 전 
model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
#eval_metric='mlogloss' 워닝을 막기 위해 로스 지정 
scores = model.score(x_test, y_test)
print("model score : ", scores)

#2번을30개 삭제한 데이터의 score model score :  0.9459459459459459P
#######################################################################
print("+++++++++++++++smoete 적용 +++++++++++++++++++++++++")

smote = SMOTE(random_state=66)
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
# # 0    53
# # 1    53
# # 2    53

# print(x_smote_train.shape,y_smote_train.shape) #(159, 13) (159,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())

# smote 전 :  (111, 13) (111,)
# smote 후 :  (159, 13) (159,)
# smote전 레이블 값 분포 :
#  1    53
# 0    44
# 2    14
# dtype: int64
# smote후 레이블 값 분포 :
#  0    53
# 1    53
# 2    53

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("model2.score : ", score)