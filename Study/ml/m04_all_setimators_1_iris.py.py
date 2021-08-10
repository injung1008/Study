
#^^ <여러모델 다 돌리기>

from sklearn.utils import all_estimators
#estimators = 추정량 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 


import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=9,stratify=y)


#! scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#!model 구성 
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
#많은 모델들이 들어있다 regressor = 회귀모델 / classifier = 분류모델
# print(allAlgorithms) 
# [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), 
#! -> 이름과 모델 형태로 들어가 있다 여러개 들어가 있는 모델은 하나씩 전부 acc를 뽑으려고 for문을 돌린다 

print('모델의 갯수 : ',len(allAlgorithms)) #41



for (name,algorithm) in allAlgorithms:
        try : 
                model = algorithm()

                model.fit(x_train, y_train)

                y_predict = model.predict(x_test)
                acc = accuracy_score(y_test, y_predict)
                print(name,'의 정답률 : ', acc)

        except : print(name, ': 없다')
        #이름 확인안할거면 pass  이름 확인하면 print 


# 모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  0.9333333333333333
# BaggingClassifier 의 정답률 :  0.9777777777777777
# BernoulliNB 의 정답률 :  0.6888888888888889      
# CalibratedClassifierCV 의 정답률 :  0.8222222222222222
# CategoricalNB : 없다
# ClassifierChain : 없다
# ComplementNB : 없다
# DecisionTreeClassifier 의 정답률 :  0.8888888888888888
# DummyClassifier 의 정답률 :  0.3333333333333333       
# ExtraTreeClassifier 의 정답률 :  0.9333333333333333   
# ExtraTreesClassifier 의 정답률 :  0.9333333333333333
# GaussianNB 의 정답률 :  0.8888888888888888
# GaussianProcessClassifier 의 정답률 :  0.8888888888888888GradientBoostingClassifier 의 정답률 :  0.9333333333333333
# HistGradientBoostingClassifier 의 정답률 :  0.9333333333333333
# KNeighborsClassifier 의 정답률 :  0.9333333333333333     
# LabelPropagation 의 정답률 :  0.9555555555555556
# LabelSpreading 의 정답률 :  0.9555555555555556
# LinearDiscriminantAnalysis 의 정답률 :  0.9777777777777777
# LinearSVC 의 정답률 :  0.9333333333333333
# LogisticRegression 의 정답률 :  0.9333333333333333       
# LogisticRegressionCV 의 정답률 :  0.9333333333333333
# MLPClassifier 의 정답률 :  0.8222222222222222
# MultiOutputClassifier : 없다
# MultinomialNB : 없다
# NearestCentroid 의 정답률 :  0.7555555555555555
# NuSVC 의 정답률 :  0.9111111111111111
# OneVsOneClassifier : 없다
# OneVsRestClassifier : 없다
# OutputCodeClassifier : 없다
# PassiveAggressiveClassifier 의 정답률 :  0.8666666666666667
# Perceptron 의 정답률 :  0.8
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9777777777777777
# RadiusNeighborsClassifier 의 정답률 :  0.8888888888888888RandomForestClassifier 의 정답률 :  0.9333333333333333
# RidgeClassifier 의 정답률 :  0.8
# RidgeClassifierCV 의 정답률 :  0.8
# SGDClassifier 의 정답률 :  0.7777777777777778
# SVC 의 정답률 :  0.9333333333333333
# StackingClassifier : 없다
# VotingClassifier : 없다
# PS D:\Study> 