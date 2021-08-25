import tensorflow as tf  
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
#<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref> 

sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa : ", aaa) # aaa :  [2.2086694]
#가중치를 랜덤한 값으로 생성한것이다 

sess.close() #세션 닫기 

#새로운 세스 만들기
sess = tf.InteractiveSession() 
#세션으로(tf.compat.v1.Session() ) 열지 않고이름만바뀐것 
sess.run(tf.global_variables_initializer())
bbb = W.eval() #세션 실현 시켜주는것 (sess.run대신)
print("bbb : ", bbb) #bbb :  [2.2086694] (aaa와 같은 값 나옴 )
sess.close()

#세션에서도 .eval사용가능 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc : ", ccc) #ccc :  [2.2086694] 셋다 같은값 
sess.close()

#같은 값이 나오지만 3가지 다른 방식으로 표현 