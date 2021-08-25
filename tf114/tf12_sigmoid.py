import tensorflow as tf 
tf.set_random_seed(66)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]      #(6,2)
y_data = [[0], [0], [0], [1], [1], [1]]      #(6,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])


w = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

# hypothesis 에서 값을 제한한다 ! 
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # 값을 0-1 사이의 값을 내보낸다 
#loss도 바이너리 크로스 엔트로피로 진행 해야함(시그모이드 친구)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))#바이너리 크로스 엔트로피 
#결과값이 마이너스로 나오기 때문에 - 붙여 줘야한다 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())

# predict = 

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0 :
        print(epochs, "cost : ", cost_val, "\n", hy_val)

#평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#조건에 대해서 true = 1, false = 0 출력 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y ), dtype=tf.float32))


c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("에측값 : \n", hy_val, 
        "\n predict : \n", c, 
        "\n Accuracy :", a)

sess.close()


