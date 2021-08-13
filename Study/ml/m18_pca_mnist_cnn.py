#실습 
#mnist데이터를 pca를 통해 cnn으로 구성 

import numpy as np     
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)


#실습 
#pca를 통해 0.95이상인 n_components 가 몇개인지? 
x = np.append(x_train, x_test, axis=0)

x = x.reshape(70000, 784)

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)



pca = PCA(n_components=400) #20 x20으로 압축 
x = pca.fit_transform(x)
# x_test = pca.fit_transform(x_test)



x_train = x[:60000]
x_test = x[60000:70000]
# print(x_train.shape)
# print(x_test.shape)

#스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
#scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train[0])




# shape늘려주기 
x_train = x_train.reshape(60000,20,20,1)
x_test = x_test.reshape(10000, 20,20,1)

# print(x_train.shape) #(60000, 20, 20, 1)
# print(x_test.shape) #(10000, 20, 20, 1)





from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
# print(y_test.shape)



# #2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(20,20,1)))
model.add(Flatten())                                           
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax')) # 왜 시그모이드를 사용할까? 
model.summary()


# #3. 컴파일 훈련 metrics = 'acc'
model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, verbose=1,
batch_size=300)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )


# #acc로만 판단  - 축소 이전 
# #metrix :  0.9714000225067139 , epochs=100
# # metrix :  0.9793000221252441

# #acc로만 판단  - 축소 이후
# loss :  [0.47372928261756897, 0.9714000225067139]
# metrix :  0.9714000225067139