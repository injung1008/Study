from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pickle


# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = pickle.load(open('./_save/xgb_save/m21_pickle.dat', 'rb'))
print('불러왔다.')

# ####################################################################################


# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], 
        eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)


# 4. 평가
results = model.score(x_test, y_test)
print('results : ', results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)