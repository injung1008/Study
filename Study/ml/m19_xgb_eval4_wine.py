from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 
import numpy as np 
from sklearn.metrics import r2_score ,accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#1. data 
datasets = load_wine() 
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
model = XGBClassifier(n_estimators=10, learning_rate=0.05, n_jobs=1) 


#3. 훈련 
model.fit(x_train, y_train, verbose=1, 
            # eval_metric=['mlogloss'],
            eval_set=[(x_train, y_train),(x_test, y_test)]) 

#eval_metric=['rmse','mae','logloss'] 3가지의 경우 결과값 
#n_estimators 숫자만큼훈련의 숫자이다 

#XGBClassifier의 기본 지표 
# [99]    validation_0-mlogloss:0.04088   validation_1-mlogloss:0.12571

#4. 평가 
result = model.score(x_test, y_test)
print('result : ', result)
#xgb의 기본 평가 셋은 rmse 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

#튜닝 후 - 최적의 파라미터값을 찾는것이 중요하다 
# result :  1.0
# acc :  1.0