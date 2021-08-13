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

#2. 모델                                                        
model = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=1) 


#3. 훈련 
model.fit(x_train, y_train, verbose=1, 
            eval_metric=['rmse']
            ,eval_set=[(x_train, y_train),(x_test, y_test)],
            early_stopping_rounds=10) 

#early_stopping_rounds=10 10번의 갱신이 없으면 멈추겠다 

#eval_metric=['rmse','mae','logloss'] 3가지의 경우 결과값 
#n_estimators 숫자만큼훈련의 숫자이다 
#[299]  
# validation_0-rmse:0.21947       validation_0-mae:0.16035   
# validation_0-logloss:-791.72473 validation_1-rmse:2.35470  
# validation_1-mae:1.77789        validation_1-logloss:-799.52997


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
#히스토리 값이랑 같음 hist = model.fit -> history.hist

# ==================================
# {'validation_0': OrderedDict([('rmse', [22.709885, 21.637075,
#  20.617416,19.64703, 18.724216, 17.848776, 17.012388, 16.220173, 
# 15.464605, 14.749583])]), 
# 'validation_1': OrderedDict([('rmse', [22.846729, 21.757334,
#  20.722433, 19.743639, 
# 18.806763, 17.932978, 17.085106, 16.280655,
#  15.507709, 14.779596])])}




# 튜닝전 
# result :  0.9220259407074536
# r2 :  0.9220259407074536

#튜닝 후 - 최적의 파라미터값을 찾는것이 중요하다 
# result :  0.9336635688400665
# r2 :  0.9336635688400665