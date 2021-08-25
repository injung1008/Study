#텐서플로우2에서도 사용가능하게 하기 파이썬 3.8.8.64.bit사용 
import tensorflow as tf
print(tf.__version__) #2.4.1

print(tf.executing_eagerly()) #True

tf.compat.v1.disable_eager_execution()
             #즉시 실행 모드 ? 

print(tf.executing_eagerly()) #False

exit()
# print('hello')

hello = tf.constant("hello world")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello)) #b'hello world'