import autokeras as ak

from tensorflow.keras.datasets import mnist

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리 - 수동으로 스케일링을 진행하고 싶을때 
x_train = x_train.reshape(-1,28*28*1)
x_test = x_test.reshape(-1,28*28*1)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
#이미지의 형식에 맞는 shape를 넣어줘야한다 3차원 또는 4차원(칼러값)또한 가능 
print(x_train.shape, x_test.shape)


#model
#이미지 분류 모델 - ImageClassifier 1차원 받지 않는다 
#회귀모델 - regressor
#수치에 대한 분류 ak.StructuredDataClassifier or 회귀모델도 있다 

#가중치 저장 부분 다 똑같고 전처리 다같음 하지만 파라미터 부분이 간소화 되어있다 

#함수형 모델이라고 생각을 하고 어느정도 튜닝이 가능한 모델 
#튜닝을 하고 싶을때 
inputs = ak.ImageInput() #함수형 모델의 input과 유사 
outputs = ak.ImageBlock(
    block_type='resnet', #모델은 resnet사용할것임 
    normalize=True, #정규화 
    augment=False
)(inputs)
outputs = ak.ClassificationHead()(outputs) #카테고리칼이라고 생각하면 된다 (카테고리칼하게 사용하겠다)-공식문서 확인하기 
#요기까지 ImageClassifier와 비슷함 

model = ak.AutoModel(
    inputs=inputs, outputs=outputs, overwrite=True, max_trials=1
)


# #간단하게 자동으로 하고싶을때 이거사용 
# model = ak.ImageClassifier(
#     overwrite=True,
#     max_trials=1, #통상적으로 2개 사용한다 - 훈련의 방법을 몇개로 진행할 것인가 
# ) #모델은 1번돌고(max_trials=1) 훈련은 2번 시킴 (epochs=2)

# #model.summary() #ImageClassifiers는 summary 제공하지 않아서 훈련을 시켜서 summary를 잡아야한다 


# 3. 컴파일 , 훈련 
model.fit(x_train, y_train, epochs=5)

#4. 평가 예측 
y_pred= model.predict(x_test)
print('y_pred' , y_pred[:10])

results = model.evaluate(x_test,y_test)
print(results)

model2 = model.export_model()
model2.summary()
#model이 여러개여서 export_model()로 정해주고 summary를 해줘야 결과가 나온다 

