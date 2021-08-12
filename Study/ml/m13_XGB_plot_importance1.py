from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from xgboost.plotting import plot_importance

#1. data

datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, 
datasets.target, train_size=0.8,random_state=66)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
model = XGBRFRegressor()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc )

                                  
print(model.feature_importances_) 


import matplotlib.pyplot as plt   
import numpy as np   

plot_importance(model) #XGBModel or dictionary여야한다 
plt.show()
#     raise ValueError('tree must be Booster, XGBModel or dict 
# instance')



# def plot_feature_importances_dataset(model):
#     n_feature = datasets.data.shape[1]
#     plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_feature), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_feature)

# plot_feature_importances_dataset(model)
# plt.show()