import numpy as np
from numpy.lib.function_base import average 
import pandas as pd 
from sklearn.datasets import load_wine 
from imblearn.over_sampling import SMOTE  
from sklearn.datasets import load_wine 
import pandas as pd 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

datasets = pd.read_csv('D:\Study\_data\winequality-white.csv',
                        index_col=None, header=0, sep=';')
# print(datasets.head)
# print(datasets.shape) #(4898, 12)
# print(datasets.describe())

datasets = datasets.values #넘파이로 변환 to_numpy사용하면 <class 'method'> 로 변함 
# print(type(datasets))

x = datasets[:,:11]
y = datasets[:,11]
# print(x.shape, y.shape) #(4898, 11) (4898,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split 


# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
# print(y)
# [6. 6. 6. ... 6. 7. 6.]



################################################
#라벨 대 통합 ! 
################################################
# print("----------------------------------------")
# print(pd.Series(y).value_counts())

#라벨 변경해서 라벨  폭 줄이기
# for i,j in enumerate(y):
#      if j == 9 : 
#          y[i] = 8
#      elif j == 3:
#          y[i] == 4

#라벨 변경해서 라벨  폭 줄이기 - for문 안돌리고 where로 변경 가능        
y = np.where(y==9,7,y)
y = np.where(y==8,7,y)
y = np.where(y==3,5,y)
y = np.where(y==4,5,y)

print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1640
# 7.0    1060

#99 - 최고치 
#98 증폭 데이터가 더 높음 
x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=98,stratify=y
)
#랜덤스테이트를 바꿀때마다 달라진다 라벨 갯수가 달라진다
# stratify=y_new 로 해놓으면 각 y의 라벨의 비율만큼 동일하게 나온다 

# print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4
# ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
#라벨값이 6개 이상이여야 증폭이 가능하다. 현재는 9번 라벨이 4개수준 

#데이터 증폭 전 
model = XGBClassifier(n_jobs=2)
model.fit(x_train, y_train, eval_metric='mlogloss')
#eval_metric='mlogloss' 워닝을 막기 위해 로스 지정 
scores = model.score(x_test, y_test)
print("전 model score : ", scores)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1 score :", f1)
# model score :  0.643265306122449

# #######################################################################
print("+++++++++++++++smoete 적용 +++++++++++++++++++++++++")

smote = SMOTE(random_state=9,k_neighbors=5)
#디폴트 neighbors값을 줄인다 , 하지만 n_neighbors값을 줄이면 연산이 줄어들어 값이 떨진다
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())


# print(x_smote_train.shape,y_smote_train.shape) #(159, 13) (159,)

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

# 전 model score :  0.7281632653061224
# +++++++++++++++smoete 적용 +++++++++++++++++++++++++
# smote 전 :  (3673, 11) (3673,)
# smote 후 :  (4944, 11) (4944,)
# smote전 레이블 값 분포 :
#  6.0    1648
# 5.0    1230
# 7.0     795
# dtype: int64
# smote후 레이블 값 분포 :
#  6.0    1648
# 5.0    1648
# 7.0    1648
# dtype: int64
# 후 model2.score :  0.7273469387755102
