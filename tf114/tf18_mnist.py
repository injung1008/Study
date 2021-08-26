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

#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000,28,28)
# x_test = x_test.reshape(10000,28,28) 
print(x_train.shape)


#2. 모델구성 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

#아웃풋 레이어 
w = tf.Variable(tf.random.normal([28*28,10]))
b = tf.Variable(tf.random.normal([1,10]))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) 


#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)


sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())

# predict = 

for epochs in range(50):
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
print('accuracy : ',a) #1.0


sess.close()



