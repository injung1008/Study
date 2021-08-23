#encoder - 암호화 
#decoder - 복호화 
#auto_encoder - 앞뒤가 똑같은 오토인코더 

import numpy as np 
from tensorflow.keras.datasets import mnist 


(x_train, _),(x_test,_) = mnist.load_data()
#값이 필요없을떄 언더바_ 하나 해주면 된다

x_train = x_train.reshape(60000, 784).astype('float')/255 
x_test = x_test.reshape(10000,784).astype('float')/255

#2. 모델
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input 

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) #연산을 줄여서 불필요한 것들은 삭제한다, 진한 특성은 남겨짐(숫자가 높은것들 - 눈코입등 )
decoded = Dense(784, activation='sigmoid')(encoded)# sigmoid가 가장 선명함
# encoded = Dense(1064, activation='relu')(input_img) 
# decoded = Dense(784, activation='tanh')(encoded)# relu, linear로 변경해도 되지만 값이 흐리고 명확한 특징 범위를 캐치하지 못함


autoencoder = Model(input_img, decoded)

# autoencoder.summary()

# _________________________________________________________________    
# Layer (type)                 Output Shape              Param #       
# =================================================================    
# input_1 (InputLayer)         [(None, 784)]             0
# _________________________________________________________________    
# dense (Dense)                (None, 64)                50240
# _________________________________________________________________    
# dense_1 (Dense)              (None, 784)               50960
# =================================================================    
# Total params: 101,200
# Trainable params: 101,200
# Non-trainable params: 0
# __________________________________________________________________

#3. 컴파일 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)
#x가 인풋이 되서 x가 나온다 - x 넣고 x와 비교한다 (집어넣은것과 나오는것이 같다)


#4. 평가, 예측 
decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt 
n = 10 
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
