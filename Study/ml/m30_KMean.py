# 지도  vs  비지도 - y 값의 유무 

from sklearn.datasets import load_iris 
from sklearn.cluster import KMeans #clustering 군집화 
import numpy as np 
import pandas as pd 
datasets = load_iris() 

#y값 없이 x값만으로 데이터프레임생성하기
irisDF = pd.DataFrame(data = datasets.data, columns=datasets.feature_names)
# print(irisDF)

#max_iter = 300 - kmean을 300번 하겠다 의미
#n_clusters=3 3개의 라벨을 뽑겠다 0,1,2의 라벨을 뽑아준다 if 5개라면 0,1,2,3,4로 라벨이 5개 생긴다
kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
kmean.fit(irisDF)

results = kmean.labels_
print('비지도 result : ', results)

# 비지도 result :  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
# 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1   
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2   
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2   
#  2 1]

# print('원래 y값 ', datasets.target)

# 원래 y값  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
# 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1   
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2   
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2   
#  2 2]

irisDF['cluster'] = kmean.labels_ #새로운 라벨값 추가 
irisDF['target'] = datasets.target #원래 y값 

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_results = irisDF.groupby(['target','cluster'])['sepal length (cm)'].count()
print(iris_results)
# target  cluster
# 0       0          50
# 1       1          48
#         2           2 틀림
# 2       1          14 틀림
#         2          36