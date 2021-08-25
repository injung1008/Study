#y = wx + b  구현 하기 
# w,b = 변수 / x, y = place holder (입력값은 p.h) / 완전한 고정값은 상수 

import tensorflow as tf     
tf.set_random_seed(66)

x_train = [1,2,3] #w = 1, b =0 인걸 우리는 알지만 머신은 모른다
y_train = [1,2,3]

W = tf.Variable([1], dtype=tf.float32, name='test') #[1] = 랜덤하게 내맘대로 넣은 초기값일뿐이다 
b = tf.Variable([1], dtype=tf.float32, name = 'test')

hypothesis = x_train * W + b #모델 구현 
#hypothesis는 f(x) = wx + b 표현 y 대신 사용한것 

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) #mse
#reduce_mean 평균값 내는것으로 mean과같다 square = 제곱 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #기본 제공해주는 옵티마이저 
train = optimizer.minimize(loss) #로스값은 제일 작은 값이 좋기 때문에 로스의 최소값을 잡아주는 옵티마이저 

#sess = tf.Session() - with문으로 대체 가능 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #변수 초기화 
    #훈련 2000번 실행한다 
    for step in range(2001):
        sess.run(train)#연산 
        if step %20 ==0: #20번 돌릴때마다 한번씩 출력 된다 step을 20으로 나눌때 나머지가 0 이면 print
            print(step, sess.run(loss), sess.run(W), sess.run(b))

#step %20 ==0 -> 왼쪽 변수에서 오른쪽값을 나눈후 나머지를 반환한다
