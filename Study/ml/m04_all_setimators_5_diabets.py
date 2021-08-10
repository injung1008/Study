

import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

datasets = load_diabetes()

x = datasets.data
y = datasets.target

#2 모델구성

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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

print('모델의 갯수 : ',len(allAlgorithms)) #41



for (name,algorithm) in allAlgorithms:
        try : 
                model = algorithm()

                model.fit(x_train, y_train)

                y_predict = model.predict(x_test)
                r2 = r2_score(y_test, y_predict)
                print(name,'의 정답률 : ', r2)

        except : print(name, ': 없다')
        #이름 확인안할거면 pass  이름 확인하면 print 


# 모델의 갯수 :  54
# ARDRegression 의 정답률 :  0.6002025719806849
# AdaBoostRegressor 의 정답률 :  0.5336273097296996
# BaggingRegressor 의 정답률 :  0.4608844072645851
# BayesianRidge 의 정답률 :  0.6031317857525211
# CCA 의 정답률 :  0.5825461541443027
# DecisionTreeRegressor 의 정답률 :  0.12298093668366705   
# DummyRegressor 의 정답률 :  -0.010307682180093813        
# ElasticNet 의 정답률 :  0.14911424482260882
# ElasticNetCV 의 정답률 :  0.5917708881624746
# ExtraTreeRegressor 의 정답률 :  -0.040412170357009725
# ExtraTreesRegressor 의 정답률 :  0.5058891611013496
# GammaRegressor 의 정답률 :  0.0920059553784578
# GaussianProcessRegressor 의 정답률 :  -12.36891212227172 
# GradientBoostingRegressor 의 정답률 :  0.4977123124979368HistGradientBoostingRegressor 의 정답률 :  0.4799971728301524
# HuberRegressor 의 정답률 :  0.5864804455357993
# IsotonicRegression : 없다
# KNeighborsRegressor 의 정답률 :  0.4853128420823485      
# KernelRidge 의 정답률 :  0.5974945257701709
# Lars 의 정답률 :  0.5900352656383722
# LarsCV 의 정답률 :  0.6007838659835665
# Lasso 의 정답률 :  0.5796130592301109
# LassoCV 의 정답률 :  0.6010792177557813
# LassoLars 의 정답률 :  0.45468470967039465
# LassoLarsCV 의 정답률 :  0.6007838659835665
# LassoLarsIC 의 정답률 :  0.602764944184214
# LinearRegression 의 정답률 :  0.5900352656383733
# LinearSVR 의 정답률 :  0.27909545802111835
# MLPRegressor 의 정답률 :  -0.6062072937223877
# MultiOutputRegressor : 없다
# MultiTaskElasticNet : 없다
# MultiTaskElasticNetCV : 없다
# MultiTaskLasso : 없다
# MultiTaskLassoCV : 없다
# NuSVR 의 정답률 :  0.1444650616814751
# OrthogonalMatchingPursuit 의 정답률 :  0.3443972776662052OrthogonalMatchingPursuitCV 의 정답률 :  0.5950203281004389
# PLSCanonical 의 정답률 :  -1.256553805294621
# PLSRegression 의 정답률 :  0.610470552100115
# PassiveAggressiveRegressor 의 정답률 :  0.5751903910130887
# PoissonRegressor 의 정답률 :  0.5912227265429173
# RANSACRegressor 의 정답률 :  0.22055497475173746
# RadiusNeighborsRegressor 의 정답률 :  0.19209707227744155RandomForestRegressor 의 정답률 :  0.49719384179857573
# RegressorChain : 없다
# Ridge 의 정답률 :  0.6019449568140334
# RidgeCV 의 정답률 :  0.6007676294744893
# SGDRegressor 의 정답률 :  0.6022150889403408
# SVR 의 정답률 :  0.16486638677450494
# StackingRegressor : 없다
# TheilSenRegressor 의 정답률 :  0.6014934679565294
# TransformedTargetRegressor 의 정답률 :  0.5900352656383733
# TweedieRegressor 의 정답률 :  0.08886348560205204        
# VotingRegressor : 없다