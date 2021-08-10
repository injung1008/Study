
from sklearn.datasets import load_boston
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


# # 완료 하시오 
x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size = 0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
# ARDRegression 의 정답률 :  0.8037449017755973
# AdaBoostRegressor 의 정답률 :  0.8666439779868449
# BaggingRegressor 의 정답률 :  0.871187088799726
# BayesianRidge 의 정답률 :  0.8048327455147766
# CCA 의 정답률 :  0.7757272685646831
# DecisionTreeRegressor 의 정답률 :  0.7160298644555407    
# DummyRegressor 의 정답률 :  -0.005227869326375867        
# ElasticNet 의 정답률 :  0.1614731669505377
# ElasticNetCV 의 정답률 :  0.8024332448684095
# ExtraTreeRegressor 의 정답률 :  0.6617926547441153
# ExtraTreesRegressor 의 정답률 :  0.8990606546714618
# GammaRegressor 의 정답률 :  0.20094407821226556
# GaussianProcessRegressor 의 정답률 :  -2.367377675992853 
# GradientBoostingRegressor 의 정답률 :  0.9140658084190632HistGradientBoostingRegressor 의 정답률 :  0.89552637425425
# HuberRegressor 의 정답률 :  0.7673572024536893
# IsotonicRegression : 없다
# KNeighborsRegressor 의 정답률 :  0.8255996350265673      
# KernelRidge 의 정답률 :  0.7312927363984302
# Lars 의 정답률 :  0.8044888426543627
# LarsCV 의 정답률 :  0.8032830033921294
# Lasso 의 정답률 :  0.23554778300892854
# LassoCV 의 정답률 :  0.8047025600429903
# LassoLars 의 정답률 :  -0.005227869326375867
# LassoLarsCV 의 정답률 :  0.8044516427844496
# LassoLarsIC 의 정답률 :  0.7983441148086403
# LinearRegression 의 정답률 :  0.8044888426543628
# LinearSVR 의 정답률 :  0.6912127880348045
# MLPRegressor 의 정답률 :  0.14915739553212082
# MultiOutputRegressor : 없다
# MultiTaskElasticNet : 없다
# MultiTaskElasticNetCV : 없다
# MultiTaskLasso : 없다
# MultiTaskLassoCV : 없다
# NuSVR 의 정답률 :  0.6350517785854072
# OrthogonalMatchingPursuit 의 정답률 :  0.5651272222459414OrthogonalMatchingPursuitCV 의 정답률 :  0.7415292549226279
# PLSCanonical 의 정답률 :  -2.2717245026237833
# PLSRegression 의 정답률 :  0.7738717095948147
# PassiveAggressiveRegressor 의 정답률 :  0.6287633552879854
# PoissonRegressor 의 정답률 :  0.6765413072129802
# RANSACRegressor 의 정답률 :  0.5746181505539568
# RadiusNeighborsRegressor 의 정답률 :  0.4099241828329552
# RandomForestRegressor 의 정답률 :  0.8850693281425843
# RegressorChain : 없다
# Ridge 의 정답률 :  0.8002975771228686
# RidgeCV 의 정답률 :  0.8048290853658459
# SGDRegressor 의 정답률 :  0.7967864967801521
# SVR 의 정답률 :  0.6677204878163618
# StackingRegressor : 없다
# TheilSenRegressor 의 정답률 :  0.7587333627965296
# TransformedTargetRegressor 의 정답률 :  0.8044888426543628
# TweedieRegressor 의 정답률 :  0.19517727399979223        
# VotingRegressor : 없다