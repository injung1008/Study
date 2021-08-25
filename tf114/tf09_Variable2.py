#실습 tf09 1번의 3가지 방식으로 출력하시오 
import tensorflow as tf 
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b 

sess = tf.compat.v1.Session() 
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print("aaa : ", aaa) # aaa :  [1.3       1.6       1.9000001]
#가중치를 랜덤한 값으로 생성한것이다 

sess.close() #세션 닫기 

#새로운 세스 만들기
sess = tf.InteractiveSession() 
#세션으로(tf.compat.v1.Session() ) 열지 않고이름만바뀐것 
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval() #세션 실현 시켜주는것 (sess.run대신)
print("bbb : ", bbb) #bbb :  [1.3       1.6       1.9000001]
sess.close()

#세션에서도 .eval사용가능 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc : ", ccc) #ccc :  [1.3       1.6       1.9000001] 셋다 같은값 
sess.close()

#같은 값이 나오지만 3가지 다른 방식으로 표현 