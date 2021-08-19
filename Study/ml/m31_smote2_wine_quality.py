import numpy as np 
import pandas as pd 
from sklearn.datasets import load_wine 
from imblearn.over_sampling import SMOTE  
from sklearn.datasets import load_wine 
import pandas as pd 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
import time
import warnings
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



x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66,stratify=y
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
model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
#eval_metric='mlogloss' 워닝을 막기 위해 로스 지정 
scores = model.score(x_test, y_test)
print("model score : ", scores)

# model score :  0.643265306122449

# #######################################################################
# print("+++++++++++++++smoete 적용 +++++++++++++++++++++++++")

smote = SMOTE(random_state=66,k_neighbors=2)
#디폴트 neighbors값을 줄인다 , 하지만 n_neighbors값을 줄이면 연산이 줄어들어 값이 떨진다
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_smote_train).value_counts())


# print(x_smote_train.shape,y_smote_train.shape) #(159, 13) (159,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())



model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print("model2.score : ", score)