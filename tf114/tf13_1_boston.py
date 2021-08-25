from sklearn import datasets
from sklearn.datasets import load_boston 
import tensorflow as tf  
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(506,1)
# print(x.shape, y.shape) #(506, 13) (506,) 
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=9)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random.normal([13,1]))
b = tf.Variable(tf.random.normal([1]))

# hypothesis 에서 값을 제한한다 ! 
hypothesis = tf.matmul(x, w) + b 

cost = tf.reduce_mean(tf.square(hypothesis - y))#mse


optimizer = tf.train.AdamOptimizer(learning_rate=0.8) #아담
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())

# predict = 

for epochs in range(5000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0 :
        print(epochs, "cost : ", cost_val, "\n", hy_val)

#평가 예측

#평가 예측

predicted = sess.run(hypothesis, feed_dict={x:x_test})
#조건에 대해서 true = 1, false = 0 출력 
from sklearn.metrics import r2_score, accuracy_score

r2 = r2_score(y_test, predicted)
# print(predicted)
print(r2)

sess.close()

#프레딕트



