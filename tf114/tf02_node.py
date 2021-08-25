import tensorflow as tf 

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3) #Tensor("Add:0", shape=(), dtype=float32)
sess = tf.Session()
print('sss.run(node3) : ', sess.run(node3)) #sss.run(node3) :  7.0
print('sss.run(node1,2) : ', sess.run([node1,node2])) #sss.run(node1,2) :  [3.0, 4.0]
#우리가 보고싶은 값은 sess.run을 통과해야한다
