import tensorflow as tf 
import numpy as np   
from keras.models import Sequential 
from keras.layers import Conv2D
tf.set_random_seed(66)

from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001 
training_epochs = 15 
batch_size = 100 
total_batsh = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28,1])
y = tf.placeholder(tf.float32, [None, 10])
#placeholder의 경우 공간을 창출해주는 것이기때문에 float32로 데이터 타입을 
#설정을 해주어야 한다 float64는 속도가 느리기 때문에 통상적 32사용 


#기존의 텐서2d의 모델 
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
# padding='same', input_shape=(28, 28,1))) 
#이것을 구현하려 한다 

#모델 구성 
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32]) #초기값 랜덤지정 하지만 이름과 쉐이프는 넣어줘야함 
#왜 쉐이프가 [3, 3, 1, 32] 일까?  shape=[kernel_size, input, output])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
#_____여기까지가 conv2d레이어 하나 ________________________
#기본적 conv2d구조 = conv2d(필터(아웃풓 노드), 커널 사이즈  stride=1, input_shape=(28,28,1))이런방식 
#x가 인풋 이라서 placeholder로 받으주면([None, 28, 28,1])) 됨 x가 4차원이면 w도 4차원이여야한다 [3, 3, 1, 32] 여기서 3,3 은 커널사이즈
#[None, 28, 28,1]) 1은 채널의 수라서 [3, 3, 1, 32]) 1은 받아주는 채널의수 1 채널 -> 1 채널 (흑백이면 1 컬러이면 3)
# 32 는필터 (아웃풋 노드) - 아웃풋의 채널이자 노드이자 필터 / - 정하기 나름 
#strides=[1,1,1,1],는 겹치는 간격이다 텐서 1은 사차원으로 받아줘야한다 
#padding='valid') 일때 28 - 커널 사이즈 + 스트라이드1일때는 1 -> 28 -3 +1 = 26이 된다 

print(w1) #shape=(3, 3, 1, 32) 
print(L1) #shape=(?, 28, 28, 32), 



# w2 = tf.Variable(tf.random.normal([3, 3, 1, 32])) #초기값 지정
# #차이점 = 초기값을 넣어주는것은 동일 하지만 get_variable은 초기값을 자동으로 넣어줌
# w3 = tf.Variable([1], dtype= tf.float32) #초기값 1로 하겠다 

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print('==========Variable(tf.random.normal)===================')
# print(np.min(sess.run(w2))) #-2.100019
# print(np.max(sess.run(w2))) #2.299845
# print(np.mean(sess.run(w2))) #0.018297732
# print(np.median(sess.run(w2))) #0.055132262
# print('===========get_variable===================')
# print(np.min(sess.run(w1))) #-0.14020838
# print(np.max(sess.run(w1))) #0.1384946
# print(np.mean(sess.run(w1))) #0.0045523066
# print(np.median(sess.run(w1))) #0.0091016665

