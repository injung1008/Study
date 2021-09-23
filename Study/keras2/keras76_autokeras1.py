import autokeras as ak

from tensorflow.keras.datasets import mnist

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#model
#이미지 분류 모델 - ImageClassifier 
#회귀모델 - regressor
#수치에 대한 분류 ak.StructuredDataClassifier or 회귀모델도 있다 

#가중치 저장 부분 다 똑같고 전처리 다같음 하지만 파라미터 부분이 간소화 되어있다 
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2, #통상적으로 2개 사용한다 - 훈련의 방법을 몇개로 진행할 것인가 
)

#3. 컴파일 , 훈련 
model.fit(x_train, y_train, epochs=5)

#4. 평가 예측 
y_pred= model.predict(x_test)

results = model.evaluate(x_test,y_test)
print(results)
