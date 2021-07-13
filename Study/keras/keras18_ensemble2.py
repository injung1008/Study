import numpy as np
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
x1 = np.transpose(x1) #100,3
x2 = np.transpose(x2) #100,3
y1 = np.array(range(1001,1101)) #(100,)
y2 = np.array(range(1901,2001)) #(100,)



from sklearn.model_selection import train_test_split
#train_test_split x1,x2,y1,y2 까지도 가능하다 
x1_train,x1_test,x2_train,x2_test,y1_train, y1_test,y2_train, y2_test = train_test_split(x1,x2, 
                                                        y1,y2, train_size = 0.7, random_state=66)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1 구성 
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu',name='dense2')(dense1)
dense3 = Dense(5, activation='relu',name='dense3')(dense2)
output1 = Dense(1,  name='output1')(dense3)

#2-2 모델1 구성 
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu',name='dense11')(input2)
dense12 = Dense(10, activation='relu',name='dense12')(dense11)
dense13 = Dense(10, activation='relu',name='dense13')(dense12)
dense14 = Dense(10, activation='relu',name='dense14')(dense13)
output2 = Dense(1, name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu', name='vvden')(merge2)

#merge3에서 갈라져서 나오게되는 y1_train
output21 = Dense(7,name='abc')(merge3)
last_output1 = Dense(1,name='abcd')(output21)
#y2_train
output22 = Dense(8)(merge3)
last_output2 = Dense(1)(output22)


#concatenate, Concatenate 소문자와 대문자의 차이는 소문자 = 메소드, 대문자- 클래스
model = Model(inputs=[input1,input2] , outputs=[last_output1,last_output2])

model.summary()

#3. 컴파일 , 훈련
#metrics=['mae', 'mse'] => 두개도 가능 
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
# x =3 개, y= 1개 이기땜시
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=20, batch_size=8, verbose=1 )

#평가예측
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test])
print(loss) 
#5개의 loss가 나오느데 첫번째는 전체loss 두번째= 1번모델 로스/ 세번째 =2번모델 로스
#두번째와 세번째의 로스를 합하면 전체 로스가 나오게 된다 
# 4,5번째 로스는 매트릭스의 로스를 말하게 된다
print('loss : ', loss[0])
print('mae : ', loss[1])

# result = model.predict([x1_test,x2_test])
# print('결과 : ', result)

# from sklearn.metrics import r2_score
# r2 = r2_score([y1_test, y2_test], result)
# print('r2 : ', r2)