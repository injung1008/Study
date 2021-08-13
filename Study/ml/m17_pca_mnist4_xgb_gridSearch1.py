# m31모델로 만든 0.999 이상의 n_component = ?를 사용하여 xgb모델을 
# 만들것 (디폴트)

# mnist dnn보다 성능 좋게 만ㄷ르어라 
# dnn , cnn과 비교 

# randomserch로도 해볼것 

import numpy as np     
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)


#실습 
#pca를 통해 0.95이상인 n_components 가 몇개인지? 
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)


x = x.reshape(70000, 784)

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)



pca = PCA(n_components=784-331) #총 10개 칼럼을 7개로 압축하겠다 
x = pca.fit_transform(x)

# 주성분 분석이 어느정도 영향을 미치는지 알고 있다면 판단 하는게 쉬워짐 차원축소의 비율에 대해서 확인함 
pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(cumsum) 

# print(np.argmax(cumsum >= 0.99)+1) #331
# import matplotlib.pyplot as plt     
# plt.plot(cumsum)
# plt.grid() 
# plt.show()

# print(x_train.shape, y_train.shape)


# print(x.shape) #(70000, 630)

x_train = x[:60000]
x_test = x[60000:70000]
y_train = y[:60000]
y_test = y[60000:70000]
# print(x_train.shape)
# print(x_test.shape)


from xgboost import XGBRegressor
from xgboost.plotting import plot_importance


model = XGBRegressor()



from sklearn.model_selection import KFold, GridSearchCV

#GridSearchCV - 모델, 러닝레이트, 크로스발리데이션 까지 한꺼번에 랩으로 싸겠다. 그리고 이 자체를 하나의 모델로 본다 
#이 모델은 기존 모델과 우리가 넣고싶은 하이퍼 파라미터의모임을 딕셔너리 형태로 구성해주고 크로스 발리데이션 까지명시해주면 모델의 형태가 된다
#크로스 발리데이션만큼 돌아간다 cv를 5번 주면 모델이 5번 돌아가고 노드가 3개면 15번 돌아가고 러닝레이트가 3이면 45번 돌아간다 


kfold = KFold(n_splits=5, shuffle=True, random_state=66)
#n_splits=5 전체 데이터를 5등분해서 5번 반복해준다 = 20% test  (등분한 수만큼 반복을 해준다) 
#cross_val_score = 교차검증 방법으로 kfold와비슷 

parameters = [
    {'n_estimators' : [100,200,300], "learning_rate":[0.1,0.3,0.001,0.01],'max_depth' : [4, 5, 6]}
    # ,{'n_estimators' : [90,100,110,200], "learning_rate":[0.1,0.001,0.01],
    # 'max_depth' : [4, 5, 6], 'colsample_bytree':[0.6,0.9,1]},
    # {'n_estimators' : [90,110], "learning_rate":[0.1,0.3,0.001,0.5],'max_depth' : [6, 8, 10, 12],
    # 'colsample_bytree':[0.6,0.9,0.9]}
]
n_jobs = -1

#!2. model 구성 
model = GridSearchCV(XGBClassifier(), parameters, cv=kfold)
#파라미터와 cv를 곱한것만큼 돌아간다 SVC에는 여러가지 파라미터가 존재한다 
#gridSearch에서는 fit을 지원한다 
# model = SVC()

#3.훈련
model.fit(x_train,y_train)
#저 파라미터중 어떤 파라미터가 가장 좋은값을 내는지 확인하는게 중요하다 
#이후 가장 좋은 파라미터를 가지고 다시 훈련을 시키면 된다 

#4. 평가, 예측 
print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 
#best_estimator_ 가장 좋은 평가가 무엇인가? 
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("best_score_ : ", model.best_score_)


print("model.score :", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_pred))


                                  

# #acc로만 판단  - 축소 이전 
# #metrix :  0.9714000225067139 , epochs=100
# # metrix :  0.9793000221252441

# #acc로만 판단  - 축소 이후 (0.95 이상들 제거 630개로 진행 )
# # loss :  [0.17470437288284302, 0.9726999998092651]
# # metrix :  0.9726999998092651

# #acc로만 판단  - 축소 이후 (0.99 이상들 제거 784-331개로 진행 )
# # loss :  [0.19386224448680878, 0.9746999740600586]
# # metrix :  0.9746999740600586

# # XGBRegressor()
# # acc :  0.8462128072548148