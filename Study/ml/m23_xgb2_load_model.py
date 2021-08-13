from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
import numpy as np 
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#1. data 
datasets = load_boston() 
x = datasets.data
y = datasets.target

print(x.shape,y.shape) #(506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(x, 
y, train_size=0.8,random_state=66)


#스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
#scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#저장 
model = XGBRegressor() #model_svae는 로드할때 모델이 무엇인지 정의해줘야한다 
model.load_model('./_save/xgb_save/m23_xgb.dat')

#4. 평가 
result = model.score(x_test, y_test)
print('result : ', result)
#xgb의 기본 평가 셋은 rmse 

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ',r2)

print('==================================')
evals_result = model.evals_result() 
print(evals_result)

# result :  0.9354279880084962
# r2 :  0.9354279880084962