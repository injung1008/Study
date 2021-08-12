from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

#1. data

datasets = load_diabetes()
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
# acc :  0.37642560129716196
# [0.06279203 0.01166884 0.27614184 0.10312547 0.04239764 0.05710048
#  0.04905772 0.02123158 0.30910992 0.06737449]

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