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
model = XGBRegressor(n_estimators=10, learning_rate=0.05, n_jobs=1) 


#3. 훈련 
model.fit(x_train, y_train, verbose=1, 
            eval_metric=["rmse", "logloss"]
            ,eval_set=[(x_train, y_train),(x_test, y_test)]) 

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
results = model.evals_result() 
print(results)
#히스토리 값이랑 같음 hist = model.fit -> history.hist

# ==================================
# {'validation_0': OrderedDict([('rmse', [22.709885, 21.637075,
#  20.617416,19.64703, 18.724216, 17.848776, 17.012388, 16.220173, 
# 15.464605, 14.749583])]), 
# 'validation_1': OrderedDict([('rmse', [22.846729, 21.757334,
#  20.722433, 19.743639, 
# 18.806763, 17.932978, 17.085106, 16.280655,
#  15.507709, 14.779596])])}


import matplotlib.pyplot as plt   
import numpy as np   


results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()

plt.ylabel('Log rmse')
plt.title('XGBoost Log rmse')
plt.show()

def Snippet_189(): 
    print()
    print(format('Hoe to visualise XGBoost model with learning curves','*^82'))    

    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from matplotlib import pyplot
    import matplotlib.pyplot as plt    
    
    plt.style.use('ggplot')    
    
    # load data
    dataset = loadtxt('pima.indians.diabetes.data.csv', delimiter=",")
    
    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
    
    # fit model no training data
    model = XGBClassifier()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
    
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    
# plot log loss
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()

plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')



# plot classification error
fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()

plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()



# 튜닝전 
# result :  0.9220259407074536
# r2 :  0.9220259407074536

#튜닝 후 - 최적의 파라미터값을 찾는것이 중요하다 
# result :  0.9336635688400665
# r2 :  0.9336635688400665