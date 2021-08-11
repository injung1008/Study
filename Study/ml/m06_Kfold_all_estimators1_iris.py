
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
from sklearn.model_selection import KFold, cross_val_score

datasets = load_iris()

x = datasets.data
y = datasets.target



#!model 구성 
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
#많은 모델들이 들어있다 regressor = 회귀모델 / classifier = 분류모델
# print(allAlgorithms) 
# [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), 
#! -> 이름과 모델 형태로 들어가 있다 여러개 들어가 있는 모델은 하나씩 전부 acc를 뽑으려고 for문을 돌린다 

print('모델의 갯수 : ',len(allAlgorithms)) #41

kfold = KFold(n_splits=5, shuffle=True, random_state=66)


for (name,algorithm) in allAlgorithms:
        try : 
                model = algorithm()

                score = cross_val_score(model,x,y,cv=kfold)
                print(name,'의 정답률 : ',round(np.mean(score), 4))

        except : print(name, ': 없다')
        #이름 확인안할거면 pass  이름 확인하면 print 



# 모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  0.8867
# BaggingClassifier 의 정답률 :  0.9533
# BernoulliNB 의 정답률 :  0.2933
# CalibratedClassifierCV 의 정답률 :  0.9133CategoricalNB 의 정답률 :  0.9333
# ClassifierChain : 없다
# ComplementNB 의 정답률 :  0.6667
# DecisionTreeClassifier 의 정답률 :  0.9467DummyClassifier 의 정답률 :  0.2933       
# ExtraTreeClassifier 의 정답률 :  0.9467   
# ExtraTreesClassifier 의 정답률 :  0.9467
# GaussianNB 의 정답률 :  0.9467
# GaussianProcessClassifier 의 정답률 :  0.96
# GradientBoostingClassifier 의 정답률 :  0.9667
# HistGradientBoostingClassifier 의 정답률 :  0.94
# KNeighborsClassifier 의 정답률 :  0.96
# LabelPropagation 의 정답률 :  0.96
# LabelSpreading 의 정답률 :  0.96
# LinearDiscriminantAnalysis 의 정답률 :  0.98
# LinearSVC 의 정답률 :  0.9667
# LogisticRegression 의 정답률 :  0.9667
# LogisticRegressionCV 의 정답률 :  0.9733
# MLPClassifier 의 정답률 :  0.9733
# MultiOutputClassifier : 없다
# MultinomialNB 의 정답률 :  0.9667
# NearestCentroid 의 정답률 :  0.9333
# NuSVC 의 정답률 :  0.9733
# OneVsOneClassifier : 없다
# OneVsRestClassifier : 없다
# OutputCodeClassifier : 없다
# PassiveAggressiveClassifier 의 정답률 :  0.8333
# Perceptron 의 정답률 :  0.78
# QuadraticDiscriminantAnalysis 의 정답률 :  0.98
# RadiusNeighborsClassifier 의 정답률 :  0.9533
# RandomForestClassifier 의 정답률 :  0.96
# RidgeClassifier 의 정답률 :  0.84
# RidgeClassifierCV 의 정답률 :  0.84
# SGDClassifier 의 정답률 :  0.9067
# SVC 의 정답률 :  0.9667
# StackingClassifier : 없다
# VotingClassifier : 없다
