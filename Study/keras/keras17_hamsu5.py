#함수형 모델의 레이어명에 대한 고찰 ! 

#1. 데이터
#같은 값의 데이터가 있어도 상관이 없다 데이터의 shape이 중요한것
import numpy as np
x = np.array([range(100), range(301, 401), range(1,101), range(100), range(401,501)])
x = np.transpose(x) #(100,5)

y = np.array([range(711,811), range(101, 201)])
y = np.transpose(y) #(100,2)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#함수형 모델은 input레이어를 명시를 해준다 

#레이어 명이 같아도 괜찮다 하지만 이럴경우 이름을 지정해줘서 summary에서 구분해주는 게 좋음
input1 = Input(shape=(5,)) #인풋레이어 구성
xx = Dense(3)(input1) #히든 레이어 구성 
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx) #아웃풋레이어 구성

#모델구성을 모양을 만든이후에 정의해줌 
model = Model(inputs=input1, outputs=output1)
model.summary()

#함수형을 쓰는 이유 모델을 지금처럼 하나만 쓰지 않고 여러가지 모델을 엮어쓰게 될 경우 = 앙상블모형
#모델 +모델을 할 경우 순차형모델은 한계가 있기 때문에 그럴때 함수형을 사용한다

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

#model.summary()

#3. 컴파일 훈련


#4. 평가예측 


