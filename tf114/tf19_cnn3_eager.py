import tensorflow as tf 
import numpy as np   
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D

tf.compat.v1.disable_eager_execution() #3.6.5       3.8.8
print(tf.executing_eagerly())          #False   ->  False
print(tf.__version__)                  #1.14.0  ->  2.4.1



# tf.set_random_seed(66)


from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255


lr = 0.0001 
training_epochs = 15 
batch_size = 100 
total_batsh = int(len(x_train)/batch_size)


x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28,1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#기존의 텐서2d의 모델 
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
# padding='same', input_shape=(28, 28,1), activation='relu')) 
#model.add(MaxPool2D()) (4,4) -> (2,2)
#이것을 구현하려 한다 

#모델 구성 
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 32]) 
#                                [kernel_size, input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L1) #same = Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
print(L1) #valid = Tensor("Relu:0", shape=(?, 26, 26, 32), dtype=float32)

print(L1_maxpool) #same/ valid - Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)


w2 = tf.compat.v1.get_variable('w2', shape=[3,3,32, 64])
#                             커널사이즈, 인풋 , 아웃풋 (1의 채널은 처음 인풋에만 입력 한다)
#L1_maxpool과 w2 연산 위에층의 아웃풋이 아래층의 인풋 이다. 
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2) #Tensor("Selu:0", shape=(?, 14, 14, 64), dtype=float32)
print(L2_maxpool)  #Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
#7x7사진이 64장 수준이됨

#레이어3
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3) #Tensor("Relu_1:0", shape=(?, 7, 7, 128), dtype=float32)
print(L3_maxpool)  #Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32) #4x4사진이 128장

#레이어4
w4 = tf.compat.v1.get_variable('w4', shape=[2,2,128, 64])
#  initializer=tf.contrib.layers.xavier_initializer() 가중치 값을 초기화 시켜서 연산 시키는것이다 
#가중치 값 규제 하는것
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4) #Tensor("LeakyRelu:0", shape=(?, 4, 4, 64), dtype=float32)
print(L4_maxpool)  #Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

#Flatten 의 개념은 하나로 쫙피는거기 떄문에 2x2x 64 이기떄문에 리쉐이프 사용하면 된다 
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64]) #2차원으로 줄이기 
print('flatten : ', L_flat) #flatten :  Tensor("Reshape:0", shape=(?, 256), dtype=float32)

#L5 - dnn진행 
w5 = tf.compat.v1.get_variable("w5", shape=[2*2*64, 64]) #[인풋 쉐이프, 아웃풋쉐잎]
b5 = tf.Variable(tf.random.normal([64]), name='bias1') #바이어스 초기값 설정
L5 = tf.matmul(L_flat, w5) + b5 #행렬계산 해주는것 (xw + b)
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5)#, keep_prob=0.2)
print(L5) #intended.Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

#L6 - dnn진행 
w6 = tf.compat.v1.get_variable("w6", shape=[64, 32]) #[인풋 쉐이프, 아웃풋쉐잎]
b6 = tf.Variable(tf.random.normal([32]), name='bias2') #바이어스 초기값 설정
L6 = tf.matmul(L5, w6) + b6 #행렬계산 해주는것 (xw + b)
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6)#, keep_prob=0.2)
print(L6) #intended.Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

#L7 - dnn + softmax 진행 
w7 = tf.compat.v1.get_variable("w7", shape=[32, 10]) #[인풋 쉐이프, 아웃풋쉐잎]
b7 = tf.Variable(tf.random.normal([10]), name='bias3') #바이어스 초기값 설정
L7 = tf.matmul(L6, w7) + b7 #행렬계산 해주는것 (xw + b)
hypothesis = tf.nn.softmax(L7)
print(hypothesis) #Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#3.컴파일 훈련 




#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())

#배치사이즈 나눠서 훈련시키기 
for epochs in range(training_epochs):
    avg_loss = 0
    for i in range(total_batsh): #몇번 돌까? 60000/100 = 600번 돈다
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        
        feed_dict = {x:batch_x, y:batch_y}
        batch_loss, _ = sess.run([loss, train], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batsh
    print('Epoch : ', '%04d' %(epochs +1), 'loss : {:.9f}'.format(avg_loss))
print("훈련 끝")

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
sess.close()

#Acc : 0.9895


