
from tensorflow.keras.layers import Dense, LSTM, Conv1D,Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(569, 30) (569,)
# print(np.unique(y)) #[0 1] 중복되는게 없는 유니크한 값

#실습 : 모델 시작 !! 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = MinMaxScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape) #(398, 30)
# print(x_test.shape)  #(171, 30)
# print(y_test.shape)  #(171,)
# print(y_train.shape)  #(398,)



#!model 구성 
#^^ <여러모델 다 돌리기>

from sklearn.utils import all_estimators
#estimators = 추정량 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 


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
# AdaBoostClassifier 의 정답률 :  0.9532163742690059
# BaggingClassifier 의 정답률 :  0.9298245614035088
# BernoulliNB 의 정답률 :  0.6432748538011696
# CalibratedClassifierCV 의 정답률 :  0.9824561403508771
# CategoricalNB : 없다
# ClassifierChain : 없다
# ComplementNB 의 정답률 :  0.8070175438596491
# DecisionTreeClassifier 의 정답률 :  0.9415204678362573
# DummyClassifier 의 정답률 :  0.6432748538011696       
# ExtraTreeClassifier 의 정답률 :  0.9298245614035088   
# ExtraTreesClassifier 의 정답률 :  0.9707602339181286
# GaussianNB 의 정답률 :  0.9473684210526315
# GaussianProcessClassifier 의 정답률 :  0.9766081871345029GradientBoostingClassifier 의 정답률 :  0.9590643274853801
# HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
# KNeighborsClassifier 의 정답률 :  0.9649122807017544
# LabelPropagation 의 정답률 :  0.9707602339181286    
# LabelSpreading 의 정답률 :  0.9707602339181286      
# LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
# LinearSVC 의 정답률 :  0.9824561403508771
# LogisticRegression 의 정답률 :  0.9766081871345029       
# LogisticRegressionCV 의 정답률 :  0.9766081871345029
# MLPClassifier 의 정답률 :  0.9824561403508771
# MultiOutputClassifier : 없다
# MultinomialNB 의 정답률 :  0.8596491228070176
# NearestCentroid 의 정답률 :  0.9415204678362573
# NuSVC 의 정답률 :  0.9590643274853801
# OneVsOneClassifier : 없다
# OneVsRestClassifier : 없다
# OutputCodeClassifier : 없다
# PassiveAggressiveClassifier 의 정답률 :  0.9532163742690059
# Perceptron 의 정답률 :  0.9298245614035088
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
# RadiusNeighborsClassifier : 없다
# RandomForestClassifier 의 정답률 :  0.9707602339181286
# RidgeClassifier 의 정답률 :  0.9532163742690059
# RidgeClassifierCV 의 정답률 :  0.9590643274853801        
# SGDClassifier 의 정답률 :  0.9649122807017544
# SVC 의 정답률 :  0.9766081871345029
# StackingClassifier : 없다
# VotingClassifier : 없다