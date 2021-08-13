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
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, 
                    # n_jobs=-8
                    tree_method='gpu_hist',
                    predictor='gpu_predictor', # cpu_predictor
                    gpu_id=0
) 


#3. 훈련 
import time 
start_time = time.time()

model.fit(x_train, y_train, verbose=1, 
            eval_metric=['rmse']
            ,eval_set=[(x_train, y_train),(x_test, y_test)]) 

print("걸린시간 : ", time.time()-start_time)

# njobs =1 걸린시간 :  9.319943904876709
# njobs =2 걸린시간 :  7.168686628341675
#njobs = 4 걸린시간 :  6.3363037109375
#njobs = 8 걸린시간 :  6.4199676513671875
#njobs = -1 걸린시간 :  6.387757301330566
#njobs = -4 걸린시간 :  6.356529235839844
#njobs = -8 걸린시간 :  6.3651347160339355

#njobs 없이 걸린시간 :  6.215440273284912
#tree_method='gpu_hist' 걸린시간 :  38.78210783004761

#tree_method='gpu_hist', gpu_id=0 걸린시간 :  39.19775032997131
