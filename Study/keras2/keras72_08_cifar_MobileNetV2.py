#실습 
# cifar 10과 cifar100으로 모델 만들것 
# trainable = True, false  
# FC로 만든것과 averate pooling으로 만든거 비교 




from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2 
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile 
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1,EfficientNetB7 


#실습 cifar10 완성 

from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import VGG16, VGG19


from tensorflow.keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)



#데이터 전처리 

#컨볼루션 데이터는 4차원으로 만들어줬지만, 표준화는 2차원 데이터만 가능하기떄문에
# #reshape을 통해서 다시 데이터차원 줄여주기 

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
#print(x_test)

from sklearn.preprocessing import StandardScaler, MinMaxScaler ,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
#scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_test)

#차원 늘려주기 
x_train = x_train.reshape(50000, 32,32, 3) 
x_test = x_test.reshape(10000, 32,32, 3) 

import tensorflow as tf 

#2. 모델링
#vgg는 이미지분류 모델이기 때문에 바로 3차원 받아버림 
mov = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))
mov.trainable=False #가중치를 동결한다 
model = Sequential()
model.add(mov)
# model.add(Flatten())
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
model.add(global_average_layer)
# model.add(Dense(125, activation='relu'))
# model.add(Dense(65, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()





# print(len(model.weights)) 
# #26(바이어스하나 레이어 하나 2개 * 13) -> 30 (트레이너블 파람은 0이 되고 26에서 레이어 2개를 늘려서 2*2 )
# print(len(model.trainable_weights)) #0 -> 4
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.01)

model.compile(loss='sparse_categorical_crossentropy'
,optimizer= optimizer, metrics=['acc'])


hist=model.fit(x_train, y_train, epochs=3, verbose=1,validation_split=0.2,
batch_size=300)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# print('훈련 acc : ', acc)
# print('훈련 val acc : ', val_acc)

# print('훈련 loss : ', loss)
# print('훈련 val loss : ', val_loss)


loss = model.evaluate(x_test, y_test)
print('시험 loss : ', loss[0])
print('시험 acc : ', loss[1] )




# #결과 출력 
# 1. cifar 10
# trainable = True, FC : loss = ? , acc= ?
# trainable = True, GAP: loss = 4.375980854034424 시험 acc :  0.10000000149011612
# trainable = False, FC : loss = ? , acc= ?
# trainable = False, GAP: 시험 loss :  2.0481693744659424 시험 acc :  0.2596000134944916

# 1. cifar 100 
# trainable = True, FC : loss = ? , acc= ?
# trainable = True, GAP: loss = ? , acc= ?
# trainable = False, FC : loss = ? , acc= ?
# trainable = False, GAP: loss = ? , acc= ?