# 다중분류 ( y가 0,1,2로 이루어져 있을때)
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #LogisticRegression은 로지스틱회귀분석 :  분류모델 
from sklearn.tree import DecisionTreeClassifier #의사결정 나무 = 분류모델과 회귀모델 이 있다
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트는 앙상블 모델이고 앙상블에는 배깅과 부스트가 있다 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

datasets = df.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]

# print(x.shape, y.shape) # (150,4) (150,)
# print(y) # y = 0,1,2

from sklearn.model_selection import train_test_split, KFold, cross_val_score
n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
#n_splits=5 전체 데이터를 5등분해서 5번 반복해준다 = 20% test  (등분한 수만큼 반복을 해준다) 
#cross_val_score = 교차검증 방법으로 kfold와비슷 


#!model 구성 

model = LinearSVC() 
#평균 ACC :  0.9667
# model = SVC()
#평균 ACC :  0.9667
# model = KNeighborsClassifier()
#평균 ACC :  0.96
# model = LogisticRegression()
#평균 ACC :  0.9667
# model = DecisionTreeClassifier()
#평균 ACC :  0.9533
# model = RandomForestClassifier()
#평균 ACC :  0.9533



#4. 훈련,평가예측

score = cross_val_score(model, x,y, cv = kfold)
print("ACC : ",score)
#print("평균 ACC : ",sum(score)/n_split)
print("평균 ACC : ",round(np.mean(score), 4)) #평균 ACC :  0.9667


#이렇게 두줄로 fit에서부터 score까지 끝남 cv =kfold 
# score자체가 평가를 하겠다는 의미 교차검증하겠다, kfold5번했기때문에 교차검증도 5번하겠다는 의미 
#cross_val_score = fit 포함 + score 포함 (kfold의 n_spilt만큼 교차검증을한다)

#! [0.96666667 0.96666667 1.         0.9        1.        ]
