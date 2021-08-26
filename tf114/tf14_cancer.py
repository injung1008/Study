#최종결론 accuracy_score 

from sklearn.datasets import load_breast_cancer
import tensorflow as tf  
tf.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569,1)
print(x_data.shape, y_data.shape) #(569, 30) (569, 1)

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7,random_state=9)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
#바이어스 없어도 상관없으니 0으로 잡아도 상관 없음 
#초기값으로 웨이트랑 바이어스가 크면 값이 잘나오지 않기 때문에 이것저것 해본 뒤에 최적값을 찾는것이 최선이다 
# w = tf.Variable(tf.random.normal([30,1])) #값이 엄청 낮고 nan값 출력 
# b = tf.Variable(tf.random.normal([1])) #0.64 수준 값은 출력된다 


# w = tf.Variable(tf.random.normal([30,1]))
# b = tf.Variable(tf.random.normal([1]))

# hypothesis 에서 값을 제한한다 ! 
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # 값을 0-1 사이의 값을 내보낸다 
#loss도 바이너리 크로스 엔트로피로 진행 해야함(시그모이드 친구)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))#바이너리 크로스 엔트로피 
#결과값이 마이너스로 나오기 때문에 - 붙여 줘야한다 

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
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
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#조건에 대해서 true = 1, false = 0 출력 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y ), dtype=tf.float32))


c, a = sess.run([predicted, accuracy], feed_dict={x:x_test, y:y_test})
print("에측값 : \n", hy_val, 
        "\n predict : \n", c, 
        "\n Accuracy :", a)

from sklearn.metrics import r2_score, accuracy_score
# accs = accuracy_score(y_test, predicted)
# print(accs)

# Accuracy : 0.90643275
sess.close()


