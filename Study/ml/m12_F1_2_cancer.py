#실습

from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
#1. data

datasets = load_breast_cancer()
print(datasets.feature_names)
x = datasets.data
y = datasets.target

# print(x)
df = pd.DataFrame(x, columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness', 'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',      
 'radius error', 'texture error', 'perimeter error' ,'area error' ,      
 'smoothness error', 'compactness error' ,'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',   
 'worst radius', 'worst texture' ,'worst perimeter', 'worst area',       
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
df = df.drop(['perimeter error'],axis=1)
# print(df)



x = df.to_numpy()
# print(x)
# print(y)

# print(type(x))
x_train, x_test, y_train, y_test = train_test_split(x, 
y, train_size=0.7,random_state=66)


#2. 모델                                                  데이터 삭제                    삭제 전 
# model = DecisionTreeClassifier(max_depth=5) 
# model = RandomForestClassifier() 
# model = GradientBoostingClassifier() 
model = XGBClassifier() 
# 모델마다 훈련을 영향을 받는 x요소의 값이 다 다르고, 얼마나 영향이 없는지에 따라 삭제 전과 후의 차이가 나타난다 



#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc )

                                  
print(model.feature_importances_) 

# acc :  0.9824561403508771
# [0.03474978 0.03467545 0.         0.00751567 0.00292065 0.00687929
#  0.01235841 0.02046652 0.         0.00054058 0.01655321 0.00162256
#  0.01760868 0.00867776 0.01414008 0.00689266 0.00062871 0.00279585
#  0.00357394 0.00587361 0.15269534 0.01300185 0.33814007 0.21980153
#  0.00525721 0.         0.00569597 0.06131698 0.00166174 0.00395583]



import matplotlib.pyplot as plt   
import numpy as np   
def plot_feature_importances_dataset(model):
    n_feature = x.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), df.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()