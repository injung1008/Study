from sklearn import datasets
from sklearn.datasets import load_boston

import autokeras as ak 
import pandas as pd 

#1. 데이터 
datasets = load_boston() 
x = datasets.data
y = datasets.target


#2. model 
# model = ak.StructuredDataClassifier #수치로된 분류모델 
#수치로 된 회귀모델 
model = ak.StructuredDataRegressor(overwrite=True, max_trials=1)

#3. 훈련 
model.fit(x,y, epochs=2, validation_split=0.2)

#4. 평가 예측 
results = model.evaluate(x,y)
print('result : ', results)
# - ETA: 1s - loss: 399.8107 - mean_squar16/16 [==============================] 
# - 0s 800us/step - loss: 529.2662 - mean_squared_error: 529.2662 - mse사용한것 
# result :  [529.2662353515625, 529.2662353515625]