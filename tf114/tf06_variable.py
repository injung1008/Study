#변수
import tensorflow as tf     
sess = tf.Session() 

x = tf.Variable([2], dtype=tf.float32, name='test')
#변수정의
init = tf.global_variables_initializer() #초기화
sess.run(init) #모든것의 실행은 sess.run 안에서 실행되는 것이라서 이것또한 여기서 실행을 시켜줘야한다
#변수 정의하고 이니셜 라이저 해주면 이후의 변수를 모두 초기화 해준다
#값이 초기화 되는게 아니고 그래프에 들어가기 적합한 구조로 변수 자료형구조의
#초기화를 해준다는 것이다. 

print('프린트 x 확인 : ',sess.run(x)) #프린트 x 확인 :  [2.]
#초기화를 안했을때 
#econditionError: Attempting to use uninitialized value test [[{{node _retval_test_0_0}}]]
#조건을 다 정해주고 시작을 해야한다
#텐서플로우 변수는 반드시 초기화를 해줘야한다 
