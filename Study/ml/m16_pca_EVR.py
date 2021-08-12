from re import T
import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn import datasets  
from sklearn.datasets import load_diabetes 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 



datasets = load_diabetes() 
x = datasets.data 
y = datasets.target   
print(x.shape, y.shape) #(442,10) (442,)

pca = PCA(n_components=8) #총 10개 칼럼을 7개로 압축하겠다 
x = pca.fit_transform(x)

#주성분 분석이 어느정도 영향을 미치는지 알고 있다면 판단 하는게 쉬워짐 차원축소의 비율에 대해서 확인함 
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
#중요한 순서대로 앞으로 밀어버린다 - 총합 0.9479436357350414  

#10개로 차원 축소를 안했을때
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832 0.00783199 0.00085605]
# 1.0
# 차원을 축소하면 가장 작은 영향을 끼치는 애가 빠져나간다 뒤에서 부터 빠져나가게 된다 

#누적값 확인하고 싶을때 
cumsum = np.cumsum(pca_EVR)
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]


print(np.argmax(cumsum >= 0.94)+1)
import matplotlib.pyplot as plt     
plt.plot(cumsum)
plt.grid() 
plt.show()



# print(x)
# print(x.shape) #(442, 7)

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7
#                 ,shuffle=True ,random_state=66)

# from xgboost import XGBRegressor
# model = XGBRegressor()

# #3. 훈련 
# model.fit(x_train,y_train) 

# #4. 평가 예측 
# results = model.score(x_test,y_test)
# print('결과 : ',results) #결과 :  0.8322826053081822 (test 분리 전 )