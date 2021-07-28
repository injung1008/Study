import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


x_train = np.load('D:\Study\_npy\k59_3_train_x.npy')
y_train = np.load('D:\Study\_npy\k59_3_train_y.npy')

x_test = np.load('D:\Study\_npy\k59_3_test_x.npy')
y_test = np.load('D:\Study\_npy\k59_3_test_y.npy')

print(y_train.shape)


#print(type(x_test[0])) #<class 'numpy.ndarray'>



#! 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#x y가 붙어있는경우 fit_generator사용하면 fit과 동일한 결과 나옴
hist = model.fit(x_train,y_train, epochs=30, steps_per_epoch=32,
                    validation_data=(x_test,y_test))
                   # validation_steps=4) 
                    #validation_steps=4 이런 파라미터가 있다 

import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


# plt.imshow(loss, 'gray')
# plt.show()

#위에것으로 시각화 할것 

#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('val acc : ', val_acc[-1])

