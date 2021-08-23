# #men , women 데이터로 모델링을 구성할것 
# #실습 1datetime A combination of a date and a time. Attributes: ()

# #실습 2 & 과제 . 본인 사진으로 predict 하시오 - 데이터 폴더 안에 테스트 셋으로 넣고 진행
# # 스크린샷으로 몇퍼센트 확률로 내가 남자인지 여자인지 크도 돌리고 보내기  

# import numpy as np    
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# #1. 데이터 불러오기 

# train_datagen = ImageDataGenerator(rescale=1./255,
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     rotation_range=5,
#                     zoom_range=1.2,
#                     shear_range=0.7,
#                     fill_mode='nearest'
#                     )

# test_datagen = ImageDataGenerator(rescale=1./255)

# #남자 1418 + 1912

# xy_train = train_datagen.flow_from_directory(
#     'D:\Study\_data\men_women',
#     target_size=(150,150), #이미지는 다 사이즈가 틀리기때문에 일률적으로 150,150으로 고정
#     batch_size=3430,
#     class_mode='binary', # data를 라벨링하듯이 파일을 2개로 놓은 다음 하나는 0 하나는 1로 사진으로 두군집을 분류한다
#     shuffle=False)

# xy_test = test_datagen.flow_from_directory(
#         'D:\\Study\\_data\\test_injung',
#         target_size=(150,150),
#         batch_size=10,
#         class_mode='binary'
# )
# # print(xy_train[0][0].shape, xy_train[0][1].shape) #(3309, 150, 150, 3) (3309,)
 
# # print(xy_test[0][1])#(10, 150, 150, 3)

# np.save('D:\Study\_npy\k59_4_train_x.npy', arr=xy_train[0][0])
# np.save('D:\Study\_npy\k59_4_train_y.npy', arr=xy_train[0][1])
# np.save('D:\Study\_npy\k59_4_test_x.npy', arr=xy_test[0][0])
# np.save('D:\Study\_npy\k59_4_test_y.npy', arr=xy_test[0][1])


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

# # print(x_train.shape)#(2316, 150, 150, 3)
# # print(y_train.shape)#(2316,)
# # print(x_test.shape)#(993, 150, 150, 3)
# # print(y_test.shape)#(993,)

# 데이터에 노이즈넣기 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
#0 - 0.1 사이의 값이 들거가게 된다 그러므로 최댓값이 1 + 0.1 = 1.1 이런식으로 1이 넘어가게 된다 
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) 
#최솟값은 무조건 0 최대값은 1로 하고 1.1, 1.001 이런것들 1넘는 애들은 다 1로한다
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D,Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D


def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(150, 150, 3),
                activation='relu', padding='same'))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    return model


model = autoEncoder(hidden_layer_size=154)

# 3. compile, train
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10, batch_size=32)

# 4. eval pred
output = model.predict(x_test_noised)

# 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), 
    (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개 무작위
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150,3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# basic 오토인코더 + 노이즈 가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150,3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#잡음 들어간거
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150,150,3), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()