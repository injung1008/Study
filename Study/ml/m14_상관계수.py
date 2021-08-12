import numpy as np      
import pandas as pd 
from sklearn.datasets import load_iris

datasets = load_iris() 
# print(datasets.keys())
##dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(datasets.target_names) 
##['setosa' 'versicolor' 'virginica']


x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(150, 4) (150,)

df = pd.DataFrame(x, columns=datasets.feature_names)
# print(df)

#y칼럼 추가 
df['Target'] = y
# print(df.head())

print("=================상관계수 히트 맵 =======================")
print(df.corr())

# =================상관계수 히트 맵 =======================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
# Target                      0.782561         -0.426658           0.949035          0.956547  1.000000

import matplotlib.pyplot as plt   
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()