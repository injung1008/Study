from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

#1. data

datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, 
datasets.target, train_size=0.8,random_state=66)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
model = RandomForestRegressor()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc )

                                  
print(model.feature_importances_) 
#  train 0.7 일때 : acc :  0.773608099740503
# [0.05506262 0.         0.         0.         0.03112124 0.60294617
#  0.         0.06521943 0.         0.         0.02064736 0.        
#  0.22500318]

# train data 0.8 일때 
# acc :  0.8774175457631728
# [0.03878833 0.         0.         0.         0.00765832 0.29639913
#  0.         0.05991689 0.         0.         0.01862509 0.
#  0.57861225]

#model = RandomForestRegressor() 일때 
# acc :  0.922499431890786
# [0.04164668 0.00123614 0.0071207  0.00093533 0.02557992 0.39974599
#  0.01232675 0.06653382 0.00307636 0.01162301 0.0154464  0.01096765
#  0.40376125]

import matplotlib.pyplot as plt   
import numpy as np   

def plot_feature_importances_dataset(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()