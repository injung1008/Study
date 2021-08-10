import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_csv('./_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

datasets = df.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]
# print(x.shape)
# print(y.shape)
# print(np.unique(y)) #7개 [3 4 5 6 7 8 9]


x_train, x_test, y_train, y_test =train_test_split(x,y,
    train_size = 0.7, shuffle = True, random_state=2)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



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
# AdaBoostClassifier 의 정답률 :  0.44761904761904764      
# BaggingClassifier 의 정답률 :  0.6278911564625851        
# BernoulliNB 의 정답률 :  0.4414965986394558
# CalibratedClassifierCV 의 정답률 :  0.5272108843537415   
# CategoricalNB : 없다
# ClassifierChain : 없다
# ComplementNB 의 정답률 :  0.38095238095238093
# DecisionTreeClassifier 의 정답률 :  0.5727891156462585   
# DummyClassifier 의 정답률 :  0.44421768707482995
# ExtraTreeClassifier 의 정답률 :  0.5843537414965987      
# ExtraTreesClassifier 의 정답률 :  0.6693877551020408     
# GaussianNB 의 정답률 :  0.43537414965986393
# GaussianProcessClassifier 의 정답률 :  0.5312925170068027GradientBoostingClassifier 의 정답률 :  0.5891156462585034
# HistGradientBoostingClassifier 의 정답률 :  0.6551020408163265
# KNeighborsClassifier 의 정답률 :  0.5231292517006803
# LabelPropagation 의 정답률 :  0.5306122448979592
# LabelSpreading 의 정답률 :  0.5251700680272109
# LinearDiscriminantAnalysis 의 정답률 :  0.5210884353741496
# LinearSVC 의 정답률 :  0.5244897959183673
# LogisticRegression 의 정답률 :  0.5292517006802722
# LogisticRegressionCV 의 정답률 :  0.5326530612244897
# MLPClassifier 의 정답률 :  0.5340136054421769
# MultiOutputClassifier : 없다
# MultinomialNB 의 정답률 :  0.4435374149659864
# NearestCentroid 의 정답률 :  0.3040816326530612
# NuSVC : 없다
# OneVsOneClassifier : 없다
# OneVsRestClassifier : 없다
# OutputCodeClassifier : 없다
# PassiveAggressiveClassifier 의 정답률 :  0.49795918367346936
# Perceptron 의 정답률 :  0.49387755102040815
# QuadraticDiscriminantAnalysis 의 정답률 :  0.47551020408163264
# RadiusNeighborsClassifier : 없다
# RandomForestClassifier 의 정답률 :  0.6605442176870748
# RidgeClassifier 의 정답률 :  0.5190476190476191
# RidgeClassifierCV 의 정답률 :  0.5170068027210885
# SGDClassifier 의 정답률 :  0.5170068027210885
# SVC 의 정답률 :  0.5482993197278911
# StackingClassifier : 없다
# VotingClassifier : 없다