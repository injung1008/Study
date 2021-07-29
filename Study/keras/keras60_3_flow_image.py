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


augment_size = 40000 #배치사이즈

randidx = np.random.randint(x_train.shape[0], size=augment_size)
                              #가져올놈         사이즈만큼 가져올놈에서 데려오겠다 

# print(x_train.shape[0]) # 60000
# print(randidx) #[53861 44590 10186 ... 32969 46320 34230] 랜덤하게 들어감 
#print(randidx.shape) #(40000,) 배치 사이즈만큼 랜덤하게 들어감 

#! <x,y 만들기 > - 새로운 40000개를 만들고 기존의 60000개와 합치면 100000개가 된다 

# 40000개 만들기
x_augmented = x_train[randidx].copy() #메모리가 공유되는걸 방지하기 위해 카피해서 진행.. 
y_augmented = y_train[randidx].copy()
#같은위치에 있는 40000개가 저장이 된다 x_augumented, y_augumented - 똑같은 이미지가 40000개 생성되어 있음 
#y값은 똑같은 라벨값이니 y값은 바뀌면 안된다 x의 데이터만 오른쪽 아래 위로 가듯 변화만 준다 
#40000장의 데이터를 약간씩 수정한다 

#^^ x_augumented - 40000개 가져오고, y = np.zeros(augment_size)  각각 하나의 데이터가 한장씩 바뀜
#^^  사만장의 데이터가 사만장으로 바뀜 한장이 다른 한장으로 다른한장이 다른한장으로 바뀜 
#^^ 이 과정에서 y값은 그대로기 때문에 바뀔 필요강 없다


#print(x_augmented)
x_augmented = x_augmented.reshape(x_augmented.shape[0],28,28,1) 
#이터러블 넘파이가 4차원을 받기때문에 쉐이프를 바꿔줘야함 
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0] 
                                #x는 [0]에 있고 y는 [1]에 있어서 마지막에 [0]을 붙임으로서 x만 뽑아줌

# print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#print(x_train.shape) #(100000, 28, 28, 1)

#실습 x_augmented 10 개와 원래 x_trian 10개를 비교하는 이미지를 출력할것 
#subplot(2,10, ?) 사용 

import matplotlib.pyplot as plt

plt.figure(figsize=(2,2))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.title('a')
    plt.imshow(x_train[i], cmap='gray')

    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.title('b')

    plt.imshow(x_augmented[i], cmap='gray')

plt.show()

