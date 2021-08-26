import numpy as np   
import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


tf.compat.v1.set_random_seed(777)

datasets = load_iris()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1,1)

# print(x_data.shape, y_data.shape) #(150, 4) (150, 1)

# from tensorflow.keras.utils import to_categorical #one hot 라이브러리
# y_data = to_categorical(y_data) #위에서 벡터로 바꿔주는 과정을 얘가 처리해줌 


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()



# print(np.unique(y_data)) #[0 1 2]
# print(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7, shuffle = True, random_state=42)



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
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#합친 버전 
# train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)


sess = tf.Session() 
sess.run(tf.global_variables_initializer())



for epochs in range(5000):
    loss_val, _ = sess.run([loss, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0 :
        print(epochs, "loss : ", loss_val)

# #평가 예측


y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
y_pred = np.argmax(y_pred, axis= 1)
y_test = np.argmax(y_test, axis= 1)

print(y_pred)
print(y_test)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))
a = sess.run([accuracy])
print('accuracy : ',a)

#accuracy :  [0.9777778]

sess.close()


