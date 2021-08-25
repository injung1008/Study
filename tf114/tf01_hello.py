import tensorflow as tf
print(tf.__version__)

# print('hello')

hello = tf.constant("hello world")#constant = 상수
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session() #생략 가능하지만 워닝을 막아준다 
print(sess.run(hello)) #b'hello world'
