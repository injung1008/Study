import numpy as np   
import tensorflow as tf 
tf.compat.v1.set_random_seed(777)


x_data = [[1,2,1,1],[1,2,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],
          [1,2,5,6],[1,6,6,6],[1,7,6,7]]    # 8,4
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]  # 8,3


x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])


w = tf.Variable(tf.random.normal([4,3]), name = 'weight') 
# (x열 , y열) -> x데이터와 (N, 4) * (4 , 3) -> (None ,3 )이 되기때문에
b = tf.Variable(tf.random.normal([1,3]), name = 'bias') #y의 출력 갯수만큼 출력 (행렬의 덧셈방법-행과 열의 갯수가 같아야함)
#더해지는건 1나인데 나가는게(y갯수가) 3개라서 1,3

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

#카테고리칼 크로스 엔트로피 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) 
# categorical_crossentropy




#이전 버전 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

#합친 버전 
# train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)


sess = tf.Session() 
sess.run(tf.global_variables_initializer())



for epochs in range(5000):
    loss_val, _ = sess.run([loss, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0 :
        print(epochs, "loss : ", loss_val)

# #평가 예측

predict = sess.run(hypothesis, feed_dict = {x:[[1,11,7,9]]})
print(predict, sess.run(tf.argmax(predict,1))) # 가장 높은 값에 1을 할당한다.

sess.close()
         
         