import tensorflow as tf 
tf.set_random_seed(66)
#        첫번   두번  세번  네번  다섯번
x1_data = [[73, 51, 65],  # (5,3)
           [92, 98, 11],
           [89, 31, 33],
           [99, 33, 100],
           [17., 66, 79]]
y_data = [[152], [185],[180],[205],[142]] #(5,1)

#compat.v1은 생략 가능하지만 워닝생성을 막기 위해 추가한것으로 본인 선택의지 
x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

#random_normal 워닝으로 random.normal로 변경 

#인풋의 열의 갯수가 가중치의 행의 갯수와 동일해야 가능
w = tf.Variable(tf.random.normal([3,1]))
#행렬 연산에 의해서 앞의 열과 뒤의 행이 같아야 값이 이뤄질수 있는데
# y = xw + b 로서 x 는 5행 3열이라서 가중치는 행이 3 이 되어야 한다
# 또한 y값이 5행 1열이기 때문에 wx + b의 값이 5행 1열을 맞춰줘야한다 
# 5행 3열 * 3행 1열 이면 -> 5행 1열이 되고 (5,1) + bias가 된다 
#shape의 기준은 x이기 때문에 가중치가 맞춰줘야한다
b = tf.Variable(tf.random.normal([1]))

# hypothesis = x * w + b  -> 이건일반용 이고 행렬용은 다른거 사용한다
hypothesis = tf.matmul(x, w) + b #행렬 연산용 
cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
#lr이 크면 값이 nan이 나온다 
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())


for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x1_data, y:y_data})
    if epochs % 10 == 0 :
        print(epochs, "cost : ", cost_val, "\n", hy_val)
sess.close()


