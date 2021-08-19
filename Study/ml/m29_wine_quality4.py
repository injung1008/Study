#실습 
#y 라벨 3,4,5,6,7,8,9 - > 총 3개로 간소화를 시킬것이다 
# 3,4 / 5,6,7/ 8,9 -> 이런식으로 분류를 한다 


import numpy as np 
import pandas as pd 
from sklearn.datasets import load_wine 

datasets = pd.read_csv('D:\Study\_data\winequality-white.csv',
                        index_col=None, header=0, sep=';')
# print(datasets.head)
# print(datasets.shape) #(4898, 12)
# print(datasets.describe())

datasets = datasets.values #넘파이로 변환 to_numpy사용하면 <class 'method'> 로 변함 

x = datasets[:,:11]
y = datasets[:,11]
# print(x.shape, y.shape) #(4898, 11) (4898,)
#y 라벨 3,4,5,6,7,8,9 - > 총 3개로 간소화를 시킬것이다 
# 3,4 / 5,6,7/ 8,9 -> 이런식으로 분류를 한다 

print(y.shape) #(4898,)
newlist = [] 
for i in list(y):
    if i<= 4 :
        newlist +=[0]
    elif i<= 7:
        newlist +=[1]
    else:
        newlist +=[2]
# print(newlist)
y= np.array(newlist) #넘파이 변환 
print(y)
print(y.shape) #(4898,)
# print(np.unique(y)) #[0 1 2]



from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,
                            random_state=33, shuffle=True)



scale = StandardScaler()
# scale = MinMaxScaler() 
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


#모델 
from xgboost import XGBClassifier
model = XGBClassifier()

#4 훈련 
model.fit(x_train, y_train)

#5. 결과확인 
score = model.score(x_test, y_test)

print("accuracy : ", score) #accuracy :  0.9520408163265306



