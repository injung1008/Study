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

#원핫인코딩을 하지않아도 구동이 가능하긴 하다
#수동 원핫인코등을 하고 싶을때 
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#ImageClassifier는 수동으로 원핫인코딩 하지 않으면 argmax까지 내부에서 해서 결과값을 준다 


#model
#이미지 분류 모델 - ImageClassifier 1차원 받지 않는다 
#회귀모델 - regressor
#수치에 대한 분류 ak.StructuredDataClassifier or 회귀모델도 있다 

#가중치 저장 부분 다 똑같고 전처리 다같음 하지만 파라미터 부분이 간소화 되어있다 
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1, #통상적으로 2개 사용한다 - 훈련의 방법을 몇개로 진행할 것인가 
) #모델은 1번돌고(max_trials=1) 훈련은 2번 시킴 (epochs=2)

#model.summary() #ImageClassifiers는 summary 제공하지 않아서 훈련을 시켜서 summary를 잡아야한다 


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


# - accuracy: 0.9892
# [0.0355086550116539, 0.9891999959945679]
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #        
# =================================================================     
# input_1 (InputLayer)         [(None, 28, 28)]          0
# _________________________________________________________________     
# cast_to_float32 (CastToFloat (None, 28, 28)            0
# _________________________________________________________________     
# expand_last_dim (ExpandLastD (None, 28, 28, 1)         0
# _________________________________________________________________     
# normalization (Normalization (None, 28, 28, 1)         3
# _________________________________________________________________     
# conv2d (Conv2D)              (None, 26, 26, 32)        320
# _________________________________________________________________     
# conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
# _________________________________________________________________     
# max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
# _________________________________________________________________     
# dropout (Dropout)            (None, 12, 12, 64)        0
# _________________________________________________________________     
# flatten (Flatten)            (None, 9216)              0
# _________________________________________________________________     
# dropout_1 (Dropout)          (None, 9216)              0
# _________________________________________________________________     
# dense (Dense)                (None, 10)                92170
# _________________________________________________________________     
# classification_head_1 (Softm (None, 10)                0
# =================================================================     
# Total params: 110,989
# Trainable params: 110,986
# Non-trainable params: 3
# _________________________________________________________________ 