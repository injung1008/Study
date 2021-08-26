import tensorflow as tf 
tf.set_random_seed(66)

#다층 퍼셉트론(딥러닝) 으로 xor을 해결한다
#xor data 둘이 같은 수이면 0 이고 둘중 하나라도 다른 수이면 1 이다 
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]  #(4,2)    
y_data = [[0], [1], [1], [0]]              #(4,1)

#2. 모델구성 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

#히든레이어 1 
w1 = tf.Variable(tf.random.normal([2,8])) #[2,8] = [데이터의 열의수, 내가 주고싶은 노드의수] 
#2 * 10 = 20 파라미터의수 
b1 = tf.Variable(tf.random.normal([8]))
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)  #새로운 x값이 탄생했다고 생각하고 layer1을 다음 x 자리에 추가


#히든레이어 2
w2 = tf.Variable(tf.random.normal([8,10])) #[8,10] = [이전노드의 열, 내가 주고싶은 노드의수]  
b2 = tf.Variable(tf.random.normal([10]))
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2) 
#                             이전의 레이어에 연결 하고 거기에 가중치와 바이어스를 연산

#아웃풋 레이어 
w3 = tf.Variable(tf.random.normal([10,1]))
b3 = tf.Variable(tf.random.normal([1]))
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3) 

#예시
#(N,2) * (2,10) = (N,10) * (10,1) = (N, 1)
# 인풋  히든레이어        아웃풋레이어 최종아웃풋


#바이너리 크로스 엔트로피 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.017)
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


