from re import T
import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_diabetes 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
datasets = load_diabetes() 
x = datasets.data 
y = datasets.target   
print(x.shape, y.shape) #(442,10) (442,)


pca = PCA(n_components=7) #총 10개 칼럼을 7개로 압축하겠다 
x = pca.fit_transform(x)

print(x)
print(x.shape) #(442, 7)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7
                ,shuffle=True ,random_state=66)

from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련 
model.fit(x_train,y_train) 

#4. 평가 예측 
results = model.score(x_test,y_test)
print('결과 : ',results) #결과 :  0.8322826053081822 (test 분리 전 )