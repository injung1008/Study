from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=False,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    rotation_range=5,
                    zoom_range=0.1,
                    shear_range=0.5,
                    fill_mode='nearest'
                    )

#train_datagen = ImageDataGenerator(rescale=1./255) # 위 아래 지금 거의 동일한 상태이다 

#! horizontal_flip 등 위의 저 파라미터들을 어떻게 사용할 것인가 


#! < 사진, 폴더로 된상태에서 땡겨올때 >
# xy_train = train_datagen.flow_from_directory(
#     'D:\Study\_data\men_women',
#     target_size=(150,150), 
#     batch_size=3430,
#     class_mode='binary', 
#     shuffle=False)

#^^  1. ImageDataGenerator 정의
#^^ 2. 파일(사진 그자체) 을 땡겨올경우 -> flow_from_directory() 사용 -> x,y 합체된상태 (튜플형태로 뭉쳐져 있다)
#^^ 3. 데이터에서 땡겨올 경우 (이미 수치화가 된 상태) -> flow() 사용 (x,y 분리되서 나옴 )

#증폭 - 좌우반전, 위치이동 등등 ( ImageDataGenerator 의 파라미터를 사용해서 증폭함 )

#!  np.tile = 배열을 반복하면서 새로운 축을 추가한다
# 신발 데이터를 x_train[0]. augument_size (100개 ) 만큼 증폭(만들려고) 한다 
#한가지 이미지를 가지고 여러가지 바뀐형태를 보여주기 위해서 사용한다 
#np.zeros(augument_size) 전체에 0을 넣어주겠다 - 이미지 증폭을 위한거기 때문에 y값은 임의로 0을 넣음 

augument_size = 100 #배치사이즈

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1), #x
    np.zeros(augument_size),                                               #y
    batch_size=augument_size, #통으로 augument_size로 짤리게 된다 
    shuffle=False
).next() #! 반환 방식이 iterator 방식으로 반환 !! / .next()  붙이면 브레이크역할을 해서 이거 한번 수행하고 다시 한번더 수행하지 않게 막아준다

#iterable 객체 - 반복 가능한 객체 (list, dict, set, str, bytes, tuple, range )
#iterator 객체 - 값을 차례대로 꺼낼 수 있는 객체입니다.
#iterator는 iterable한 객체를 내장함수 또는 iterable객체의 메소드로 객체를 생성할 수 있습니다.
#리스트 형태로 하나하나 반환함 - 배치사이즈 크기만큼만 반환을 한다 
#.next() 를 사용하면 하나하나 나가는게 아니고 한꺼번에(전체가) 다 나간다 

# print(type(x_data)) #<class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
#                     # .next() = <class 'tuple'>
# print(type(x_data[0])) #<class 'tuple'> -> .next 후 <class 'numpy.ndarray'>
# print(type(x_data[0][0])) #<class 'numpy.ndarray'> -> <class 'numpy.ndarray'>
# print(x_data[0][0].shape) #(100, 28, 28, 1) = x  .next 후 -> (29, 28, 1) 
# print(x_data[0][1].shape) #(100,) -= y
# #! .next 후  shape가 밀렸다 
# print(x_data[0].shape) #(100,28,28,1) = x
# print(x_data[1].shape) #(100,) = y

import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()

