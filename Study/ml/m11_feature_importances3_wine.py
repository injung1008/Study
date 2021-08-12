from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#1. data
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, 
datasets.target, train_size=0.7,random_state=66)

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc )

                                  
print(model.feature_importances_) 
# acc :  1.0
# [0.1367278  0.04015412 0.01585278 0.03855186 0.02548153 0.04879147
#  0.13803146 0.01402343 0.03123101 0.1168438  0.08533402 0.17178974
#  0.13718698]

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