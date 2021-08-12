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



pca = PCA(n_components=784-331) #총 10개 칼럼을 7개로 압축하겠다 
x = pca.fit_transform(x)

# 주성분 분석이 어느정도 영향을 미치는지 알고 있다면 판단 하는게 쉬워짐 차원축소의 비율에 대해서 확인함 
pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# print(cumsum) 

print(np.argmax(cumsum >= 0.99)+1) #331
# import matplotlib.pyplot as plt     
# plt.plot(cumsum)
# plt.grid() 
# plt.show()

# print(x_train.shape, y_train.shape)


# print(x.shape) #(70000, 630)

x_train = x[:60000]
x_test = x[60000:70000]
print(x_train.shape)
print(x_test.shape)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
print(y_test.shape)



# #2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D

model = Sequential()
model.add(Dense(512, input_dim=784-331))                                        
model.add(Dense(251, activation='relu'))
model.add(Dense(135, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 


# #3. 컴파일 훈련 metrics = 'acc'
model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=100, verbose=1, validation_split=0.025,
batch_size=300)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )



#acc로만 판단  - 축소 이전 
#metrix :  0.9714000225067139 , epochs=100
# metrix :  0.9793000221252441

#acc로만 판단  - 축소 이후 (0.95 이상들 제거 630개로 진행 )
# loss :  [0.17470437288284302, 0.9726999998092651]
# metrix :  0.9726999998092651

#acc로만 판단  - 축소 이후 (0.99 이상들 제거 784-331개로 진행 )
# loss :  [0.19386224448680878, 0.9746999740600586]
# metrix :  0.9746999740600586