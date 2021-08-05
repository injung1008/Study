import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.load('D:\Study\_npy\k59_6_train_x.npy')
y = np.load('D:\Study\_npy\k59_6_train_y.npy')

print(np.unique(y)) #[0. 1.]

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.8, shuffle=True, random_state=9)

# print(x_train.shape)#(821, 150, 150, 3)
# print(y_train.shape)#(821,)
# print(x_test.shape)#(206, 150, 150, 3)
# print(y_test.shape)#(206,)


#! 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D,Dense, Conv2D, Flatten, Dropout

import tensorflow as tf
VGG16_MODEL = tf.keras.applications.VGG16(input_shape = (150,150,3),
                                         include_top = False,
                                         weights = 'imagenet')

VGG16_MODEL.trainable=False
# flatten이 없음 ( globalaveragepooling으로 대체 )
#  ==> 가중치가 필요없음
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# FFNN의 가중치는 학습됨
prediction_layer = tf.keras.layers.Dense(3, activation ='softmax' )

model = tf.keras.Sequential([
    VGG16_MODEL,
    global_average_layer,
    prediction_layer
])
# model : vgg16이 갖고있는 가중치 + FFNN 가중치로 학습


model.summary()

# #3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3) # 


#x y가 붙어있는경우 fit_generator사용하면 fit과 동일한 결과 나옴
hist = model.fit(x_train,y_train, epochs=100,batch_size=100,
                    validation_split=0.2)
                   # validation_steps=4) 
                    #validation_steps=4 이런 파라미터가 있다 


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


loss = model.evaluate(x_test, y_test)
print('loss : ', loss )

#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('val acc : ', val_acc[-1])


temp = model.predict(x_test)
print('원본 : ', temp[:5])
temp = tf.argmax(temp, axis=1)
#temp = pd.DataFrame(temp)
print('예측값 : ', temp[:5])
print('원래값 : ',y_test[:5])

# loss :  [0.43910643458366394, 0.737864077091217]
# acc :  0.8643292784690857
# val acc :  0.8121212124824524
# 원본 :  [[1.3798817e-01 8.6192614e-01 8.5691900e-05]
#  [6.5781906e-02 9.3418962e-01 2.8542949e-05]
#  [3.4763971e-01 6.5227288e-01 8.7488334e-05]
#  [2.3917644e-03 9.9760556e-01 2.6095156e-06]
#  [2.3298435e-01 7.6636440e-01 6.5119105e-04]]
# 예측값 :  tf.Tensor([1 1 1 1 1], shape=(5,), dtype=int64)      
# 원래값 :  [1. 1. 0. 1. 1.]