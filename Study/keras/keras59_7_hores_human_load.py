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

model = Sequential()
model.add(Conv2D(100,(2,2), input_shape=(150,150,3)))
model.add(Conv2D(50, (2,2)))           
model.add(Flatten())
# model.add(Dense(160,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(22,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2, activation='softmax'))

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