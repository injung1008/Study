from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1. data
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, 
datasets.target, train_size=0.7,random_state=66)

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
model = XGBClassifier()
# model = GradientBoostingClassifier()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc )

                                  
print(model.feature_importances_) 
# acc :  0.9649122807017544
# [0.02648659 0.01581667 0.05257453 0.04595535 0.00510997 0.01285458
#  0.03546237 0.08565853 0.0029372  0.00279022 0.02579214 0.00610739
#  0.01552897 0.04339505 0.00443487 0.0057198  0.00350958 0.00359472
#  0.00477386 0.00491765 0.15466869 0.02471674 0.12232669 0.14986809
#  0.01185862 0.01380388 0.02622723 0.0803491  0.00511405 0.00764688]

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