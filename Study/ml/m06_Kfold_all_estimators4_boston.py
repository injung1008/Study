
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target




#실습, 모델 구성하고 완료하시오
#회귀 데이터를 classfier로 만들었을 경우에 에러 확인 !! 


#!model 구성 
#^^ <여러모델 다 돌리기>

from sklearn.utils import all_estimators
#estimators = 추정량 
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 


# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
#많은 모델들이 들어있다 regressor = 회귀모델 / classifier = 분류모델
# print(allAlgorithms) 
# [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), 
#! -> 이름과 모델 형태로 들어가 있다 여러개 들어가 있는 모델은 하나씩 전부 acc를 뽑으려고 for문을 돌린다 

from sklearn.model_selection import KFold, cross_val_score


kfold = KFold(n_splits=5, shuffle=True, random_state=66)


for (name,algorithm) in allAlgorithms:
        try : 
                model = algorithm()

                score = cross_val_score(model,x,y,cv=kfold)
                print(name,'의 정답률 : ',round(np.mean(score), 4))

        except : print(name, ': 없다')
        #이름 확인안할거면 pass  이름 확인하면 print 
