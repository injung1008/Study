#MCPO 파일 명 조작하기 

import numpy as np
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
x1 = np.transpose(x1) #100,3
x2 = np.transpose(x2) #100,3
y1 = np.array(range(1001,1101)) #(100,)



from sklearn.model_selection import train_test_split
#train_test_split x1,x2,y1,y2 까지도 가능하다 
x1_train,x1_test,x2_train,x2_test,y1_train, y1_test = train_test_split(x1,x2, y1, train_size = 0.7, shuffle=True, 
                                                        random_state=66)



#2. 모델구성
from tensorflow.keras.models import Model, load_model
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
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)
#concatenate, Concatenate 소문자와 대문자의 차이는 소문자 = 메소드, 대문자- 클래스
model = Model(inputs=[input1,input2] , outputs=last_output)

model.summary()

#3. 컴파일 , 훈련
#metrics=['mae', 'mse'] => 두개도 가능 
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
# x =3 개, y= 1개 이기땜시

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                    restore_best_weights=True)
##############################################################################                    
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/ModelCheckPoint'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
#filename = epoch값과 loss값이 파일명에 나올것이다 
modelpath = "".join([filepath, "k47_", date_time, "_", filename])

#체크포인트가 갱신될때마다 파일이 생성이 된다 
#실질적으로 맨 마지막이 가장 높다
################################################################################3

mcp = ModelCheckpoint(monitor = 'val_loss', mode='auto', batch_size = 8,verbose=1,
                        filepath = modelpath)    

model.fit([x1_train,x2_train], y1_train, epochs=200, batch_size=8, verbose=1,
                validation_split=0.2 ,callbacks=[es,mcp] )

model.save('./_save/ModelCheckPoint/keras40_model_.h5')
from sklearn.metrics import r2_score



print("==============기본 출력 ==============================")

#평가예측
loss = model.evaluate([x1_test,x2_test], y1_test)
print('loss : ', loss[0])


result = model.predict([x1_test,x2_test])


r2 = r2_score(y1_test, result)
print('r2 : ', r2)

print("===================1. load model ==============================")
#브레이크 잡고 20번 스탑한 지역에서 저장한것
model2 = load_model('./_save/ModelCheckPoint/keras40_model_.h5')
loss = model2.evaluate([x1_test,x2_test], y1_test)
print('loss : ', loss[0])


result = model2.predict([x1_test,x2_test])


r2 = r2_score(y1_test, result)
print('r2 : ', r2)


# print("===================2. MCP ==============================")

# model3 = load_model('./_save/ModelCheckPoint/keras49_mcp.h5')
# loss = model3.evaluate([x1_test,x2_test], y1_test)
# print('loss : ', loss[0])


# result = model3.predict([x1_test,x2_test])


# r2 = r2_score(y1_test, result)
# print('r2 : ', r2)


'''
<restore_best_weights=False>
==============기본 출력 ==============================
loss :  0.02935076504945755
r2 :  0.999966432569837
===================1. load model ==============================
loss :  0.02935076504945755
r2 :  0.999966432569837
===================2. MCP ==============================
loss :  0.02935076504945755
r2 :  0.999966432569837


<restore_best_weights=TRUE>

==============기본 출력 ==============================
loss :  1045.940673828125
r2 :  -0.1962053441871363
===================1. load model ==============================
loss :  1045.940673828125
r2 :  -0.1962053441871363
===================2. MCP ==============================
loss :  1045.940673828125
r2 :  -0.1962053441871363
'''