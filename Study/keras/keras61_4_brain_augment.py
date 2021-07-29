import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


x_train = np.load('D:\Study\_npy\k59_3_train_x.npy')
y_train = np.load('D:\Study\_npy\k59_3_train_y.npy')

x_test = np.load('D:\Study\_npy\k59_3_test_x.npy')
y_test = np.load('D:\Study\_npy\k59_3_test_y.npy')

# print(y_train.shape) #(160,)
# print(x_train.shape) #(160, 150, 150, 3)

augment_size = 32 #배치사이즈

randidx = np.random.randint(x_train.shape[0], size=augment_size)


# print(randidx) #[ 51   6   5  48 113 124  78  72  49  71   7  91 159  84   8  69  62  60
# #  99 125 133  93  62 139 155 133 155  29  20 140 129  18]랜덤하게 들어감 
# print(randidx.shape) #(32,)배치 사이즈만큼 랜덤하게 들어감 


x_augmented = x_train[randidx].copy() #메모리가 공유되는걸 방지하기 위해 카피해서 진행.. 
y_augmented = y_train[randidx].copy()
#print(x_augmented.shape) #(32, 150, 150, 3)

x_augmented = x_augmented.reshape(x_augmented.shape[0],150,150,3)  #4차원으로 만들어주기 
# #이터러블 넘파이가 4차원을 받기때문에 쉐이프를 바꿔줘야함 
x_train = x_train.reshape(x_train.shape[0], 150,150,3)
x_test = x_test.reshape(x_test.shape[0], 150,150,3)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape) #(10000, 28, 28, 3)

