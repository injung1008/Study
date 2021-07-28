import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.load('D:\Study\_npy\k59_4_train_x.npy')
y = np.load('D:\Study\_npy\k59_4_train_y.npy')

test_test = np.load('D:\Study\_npy\k59_4_test_x.npy')
y_test = np.load('D:\Study\_npy\k59_4_test_y.npy')



# print(x.shape, y.shape) #(3309, 150, 150, 3) (3309,)


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size = 0.7, shuffle=True, random_state=66)

# print(x_train.shape)#(2316, 150, 150, 3)
# print(y_train.shape)#(2316,)
# print(x_test.shape)#(993, 150, 150, 3)
# print(y_test.shape)#(993,)


#! 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(100,(2,2), input_shape=(150,150,3)))
# model.add(Conv2D(180, (2,2)))
model.add(Flatten())
# model.add(Dense(160,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#x y가 붙어있는경우 fit_generator사용하면 fit과 동일한 결과 나옴
hist = model.fit(x_train,y_train, epochs=100, steps_per_epoch=32,batch_size=100,
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


temp = model.predict(test_test)
print(temp)

# temp2 = tf.argmax(temp, axis=1)
# print(temp2)

# temp3 = pd.DataFrame(temp2)
# print(temp3)

# print(temp)


#!############################################################
#loss :  [2.4865024089813232, 0.5629405975341797]
# acc :  0.9920844435691833
# val acc :  0.984455943107605
# [[0.9923161 ]
#  [0.7146235 ]
#  [0.9984805 ]
#  [0.9998901 ]
#  [1.        ]
#  [0.94719756]
#  [0.04927225]
#  [0.9895536 ]
#  [0.9999907 ]
#  [0.99805236]]

# model = Sequential()
# model.add(Conv2D(100,(2,2), input_shape=(150,150,3)))
# # model.add(Conv2D(180, (2,2)))
# model.add(Flatten())
# # model.add(Dense(160,activation='relu'))
# model.add(Dense(80,activation='relu'))
# model.add(Dense(60,activation='relu'))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(1, activation='sigmoid'))