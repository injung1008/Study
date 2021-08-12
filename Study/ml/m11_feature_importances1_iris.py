from numpy.lib.npyio import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. data
datasets = load_iris()
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

                                  
print(model.feature_importances_) #[0.0125026  0.         0.53835801 0.44913938] =1
# 4 = 아아리스의 칼럼 갯수 이며, 4개의 칼럼의 이 훈련에 대한 영향도  0이면 아예 도움이 안되고 있음 삭제해도 무방 
# 하지만 상대적이다 train_size=0.8,random_state=66) 이 데이터에 한해서 의사결정 모델을 적용 했을때 두번재는 쓸모가 없다는 의미 
#! [0.         0.01906837 0.04351141 0.93742021]     train_size=0.7 인경우 첫번째 컬럼이 필요가없어짐 acc :  0.9111111111111111 도 떨어짐     
#DecisionTreeClassifier(max_depth=4)이런식으로 파라미터 추가해도 값이 바뀐다 

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