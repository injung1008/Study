import numpy as np 
from tensorflow.keras.datasets import mnist 
from tensorflow.python.keras import activations

(x_train, _),(x_test,_) = mnist.load_data()
#값이 필요없을때 언더바_ 하나 해주면 된다
#1. 데이터
x_train = x_train.reshape(60000, 784).astype('float')/255  #정규화 한상태 
x_test = x_test.reshape(10000,784).astype('float')/255

# 데이터에 노이즈넣기 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
#0 - 0.1 사이의 값이 들거가게 된다 그러므로 최댓값이 1 + 0.1 = 1.1 이런식으로 1이 넘어가게 된다 
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) 
#최솟값은 무조건 0 최대값은 1로 하고 1.1, 1.001 이런것들 1넘는 애들은 다 1로한다
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

#2. 모델
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input 

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
        activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154) #pca 95%

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt 
import random

fig, ((ax1, ax2, ax3, ax4, ax5),
    (ax11, ax12, ax13, ax14, ax15), 
    (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3,5, figsize=(20,7))


# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

#원본(입력) 이미지를 맨위에 그린다 
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
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


#오통니코더가 출력한 이미지를 아래에 그린다 
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
