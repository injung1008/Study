#실습 


import numpy as np
from tensorflow.keras.datasets import mnist

#1. data
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32")/255.
x_train2 = x_train.reshape(60000, 28*28).astype("float32")/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32")/255.

# 데이터에 노이즈넣기 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
#0 - 0.1 사이의 값이 들거가게 된다 그러므로 최댓값이 1 + 0.1 = 1.1 이런식으로 1이 넘어가게 된다 
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) 
#최솟값은 무조건 0 최대값은 1로 하고 1.1, 1.001 이런것들 1넘는 애들은 다 1로한다
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D

def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(28, 28, 1),
                activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))

    return model

def autoEncoderD(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(28, 28, 1),
                activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model

model1 = autoEncoder(hidden_layer_size=154)
model2 = autoEncoderD(hidden_layer_size=154)

# 3. compile, train
model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')

model1.fit(x_train_noised, x_train2, epochs=10, batch_size=32)
model2.fit(x_train_noised, x_train2, epochs=10, batch_size=32)

# 4. eval pred
output1 = model1.predict(x_test_noised)
output2 = model2.predict(x_test_noised)

# 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), 
    (ax11, ax12, ax13, ax14, ax15),(ax16, ax17, ax18, ax19, ax20)) = \
    plt.subplots(4, 5, figsize = (20, 7))

# 이미지 다섯 개 무작위
random_images = random.sample(range(output1.shape[0]), 5)
random_images = random.sample(range(output2.shape[0]), 5)

# 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# basic 오토인코더 + 노이즈 가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# deep 오토인코더 + 노이즈 가 출력한 이미지
for i, ax in enumerate([ax16, ax17, ax18, ax19, ax20]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#잡음 들어간거
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()