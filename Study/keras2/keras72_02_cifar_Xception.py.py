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
# print(x_test)

#차원 늘려주기 
# x_train = x_train.reshape(5120, 100,100, 3) 
# x_test = x_test.reshape(1024, 100,100, 3) 
x_train = x_train.reshape(3125, 32*4,32*4, 3) 
x_test = x_test.reshape(625, 128,128, 3) 
y_test = y_test.reshape(625,16)
y_train = y_train.reshape(3125,16)

print(x_train.shape,y_train.shape) #(3125, 128, 128, 3) (3125, 16)
print(x_test.shape,y_test.shape) #(625, 128, 128, 3) (625, 16)

import tensorflow as tf 
print(y_test)
# <Xception_shape>
# input_shape : 선택적 모양 튜플, include_top가 False인 경우에만 지정됩니다 
# (그렇지 않으면 입력 모양은 (299, 299, 3)이어야 합니다 . 
# 정확히 3개의 입력 채널이 있어야 하고 너비와 높이가 71보다 작아서는 안 됩니다.
#  예를 들어 (150, 150, 3)하나의 유효한 값이 될 것입니다.
# 2. 모델링

xcept = Xception(weights='imagenet', include_top=False, input_shape=(128,128,3))
xcept.trainable=True #
model = Sequential()
model.add(xcept)
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
# trainable = True, GAP: loss = 2.307157516479492 , acc= 0.10000000149011612
# trainable = False, FC : loss = ? , acc= ?
# trainable = False, GAP: loss = 1.2843854427337646 , acc= 0.5540000200271606

# 1. cifar 100 
# trainable = True, FC : loss = ? , acc= ?
# trainable = True, GAP: loss = ? , acc= ?
# trainable = False, FC : loss = ? , acc= ?
# trainable = False, GAP: loss = ? , acc= ?