#실습 
#다층레이어 구성 

#여러가지 activation 구성 가능 
# hypothesis = tf.nn.relu(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.elu(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.selu(tf.matmul(x_train, w3) + b3)
# hypothesis = tf.sigmoid(tf.matmul(x_train, w3) + b3) 
# hypothesis = tf.nn.dropout(layer1, keep_prob=0.3) 
# hypothesis = tf.nn.softmax(tf.matmul(x_train, w3) + b3) 


#실습 
# from tensorflow.keras.datasets import mnist #기존 방식 
from keras.datasets import mnist 
import tensorflow as tf 
import numpy as np
tf.set_random_seed(66)

(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1) 

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) - 얘는 4차원을 받는데 3차원이라서 차원을 늘려야한다 
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

#train만 원핫인코더 해줌 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()


print(y_train.shape) #(60000, 10)
print(y_test[:5])

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28) 

#스케일링
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
# scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)




#모델구성 
#2. 모델구성 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

#히든레이어 1 
w1 = tf.Variable(tf.random.normal([28*28,256], 0, 3e-2)) 
#0 = 평균 ,3e-2= 표준편차 지정해준것 , 값이 작을 수록 성능이 좋은것은 지금 x의 값이 0.0001정도로
#값이 매우 작은 수준이다. 또한 y 도 0 또는 1을 구분하도록 카테고리칼을 해놔서 값이 작은 상태 이상황에서 초기값으로
#설정해놓은 웨이트와 바이어스가 커져 버리면 y값에서 벗어나는 폭이 크기 때문에 아예 훈련을 하더라도 nan값으로 
#큰폭에서 부터 실행을 해버린다. 이에 웨이트의 초기값 설정 또한 스케일 즉 x와 y에 맞는 크기를 고려하여 
# 웨이트를 설정하는 것이 중요한것이다. 

#[데이터의 열의수, 내가 주고싶은 노드의수] 
# stddev는 랜덤수의 범위를 정해주는 것. default는 1.0이다 0.1로 해주면 적은 범위의 수들이 랜덤하게 뽑힌다.
# 초반의 weight를 잡기위해서 stddev를 집어넣은것.
b1 = tf.Variable(tf.random.normal([256], 0, 1e-2))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)  #새로운 x값이 탄생했다고 생각하고 layer1을 다음 x 자리에 추가
layer1 = tf.nn.dropout(layer1, keep_prob=0.1) 


#히든레이어 2
w2 = tf.Variable(tf.random.normal([256,124], 0, 3e-2)) #[8,10] = [이전노드의 열, 내가 주고싶은 노드의수]  
b2 = tf.Variable(tf.random.normal([124], 0, 1e-2))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2) 
# layer2 = tf.nn.dropout(layer2, keep_prob=0.1) 

#히든레이어 3
w3 = tf.Variable(tf.random.normal([124,62], 0, 3e-2)) #[8,10] = [이전노드의 열, 내가 주고싶은 노드의수]  
b3 = tf.Variable(tf.random.normal([62], 0, 1e-2))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3) 
#                             

#아웃풋 레이어 
w4 = tf.Variable(tf.random.normal([62,10], 0, 3e-2))
b4 = tf.Variable(tf.random.normal([10], 0, 1e-2))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4) 


#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
train = optimizer.minimize(loss)


sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())

# predict = 

for epochs in range(1200):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0 :
        print(epochs, "loss : ", cost_val, "\n", hy_val)

#평가 예측
y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_pred)
print(y_test)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))
a = sess.run([accuracy])
print('accuracy : ',a)


sess.close()



