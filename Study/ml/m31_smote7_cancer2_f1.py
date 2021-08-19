import numpy as np
from numpy.lib.function_base import average 
import pandas as pd 
from sklearn.datasets import load_wine 
from imblearn.over_sampling import SMOTE  
from sklearn.datasets import load_wine,load_breast_cancer
import pandas as pd 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target



# print(x.shape, y.shape) #(569, 30) (569,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split 

print(pd.Series(y).value_counts())
# 1    357
# 0    212

count = 0
lists = []
for i,j in enumerate(y):
    if j == 0:
        lists.append(i)
        
        count += 1
        
    elif count == 100 : break
# print(lists)
key =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 47, 53, 54, 56, 
 57, 62, 64, 65, 70, 72, 73, 75, 77, 78, 82, 83, 85, 86, 87, 91,
  94, 95, 99, 100, 105, 108, 117, 118, 119, 121, 122, 126, 127, 
  129, 131, 132, 134, 135, 138, 141, 146, 156, 161, 162, 164, 167, 
  168, 171, 172, 177, 180, 181, 182, 184, 186, 190, 193, 194]

y = np.delete(y, key)
x = np.delete(x, key,0)
# print(x[0])
# print(pd.Series(y).value_counts())
# print(x[0])
print(x.shape, y.shape) #(469, 30) (469,)



x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=98,stratify=y
)
# #랜덤스테이트를 바꿀때마다 달라진다 라벨 갯수가 달라진다
# # stratify=y_new 로 해놓으면 각 y의 라벨의 비율만큼 동일하게 나온다 

print(pd.Series(y_train).value_counts())

#데이터 증폭 전 
model = XGBClassifier(n_jobs=2)
model.fit(x_train, y_train, eval_metric='mlogloss')
#eval_metric='mlogloss' 워닝을 막기 위해 로스 지정 
scores = model.score(x_test, y_test)
print("전 model score : ", scores)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 score :", f1)

# #######################################################################
print("+++++++++++++++smoete 적용 +++++++++++++++++++++++++")

smote = SMOTE(random_state=9,k_neighbors=5)
#디폴트 neighbors값을 줄인다 , 하지만 n_neighbors값을 줄이면 연산이 줄어들어 값이 떨진다
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_smote_train).value_counts())

# ++++++++++++++적용 전 +++++++++++++++++++++++++++++
# 1    267    
# 0    159    
# dtype: int64
# +++++++++++++++smoete 적용 +++++++++++++++++++++++++
# 0    267
# 1    267

# print(x_smote_train.shape,y_smote_train.shape)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())



model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("후 model2.score : ", score)

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("후 f1 score :", f1)

# 전 model score :  0.9830508474576272
# f1 score : 0.9765873015873017
# +++++++++++++++smoete 적용 +++++++++++++++++++++++++
# 0    267
# 1    267
# dtype: int64
# smote 전 :  (351, 30) (351,)
# smote 후 :  (534, 30) (534,)
# smote전 레이블 값 분포 :
#  1    267
# 0     84
# dtype: int64
# smote후 레이블 값 분포 :
#  0    267
# 1    267
# dtype: int64
# 후 model2.score :  0.9830508474576272
# 후 f1 score : 0.9765873015873017