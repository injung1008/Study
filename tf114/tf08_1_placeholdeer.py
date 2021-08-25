#y = wx + b  구현 하기 
# w,b = 변수 / x, y = place holder (입력값은 p.h) / 완전한 고정값은 상수 

import tensorflow as tf     
tf.compat.v1.set_random_seed(66)

# x_train = [1,2,3] #w = 1, b =0 인걸 우리는 알지만 머신은 모른다
# y_train = [1,2,3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None]) #자리창출하지 어떤값을 갖고 있는지 모름 
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable([1], dtype=tf.float32, name='test') #[1] = 랜덤하게 내맘대로 넣은 초기값일뿐이다 
# b = tf.Variable([1], dtype=tf.float32, name = 'test') #초기값

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='test') #[1] = 랜덤하게 내맘대로 넣은 초기값일뿐이다 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name = 'test') #초기값



hypothesis = x_train * W + b #모델 구현 
#hypothesis는 f(x) = wx + b 표현 y 대신 사용한것 

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) #mse
#reduce_mean 평균값 내는것으로 mean과같다 square = 제곱 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #기본 제공해주는 옵티마이저 
train = optimizer.minimize(loss) #로스값은 제일 작은 값이 좋기 때문에 로스의 최소값을 잡아주는 옵티마이저 

sess = tf.Session() 

sess.run(tf.global_variables_initializer()) #변수 초기화 
#훈련 2000번 실행한다 
for step in range(2001):
    # sess.run(train)#연산
    _, loss_val, W_val, b_val= sess.run([train, loss, W, b], 
            feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
#loss결과값을 보고싶으면 run을 시켜줘야하고, 하지만 train은 훈련 자체로서 값을 볼 필요가 없기 떄문에 _ 라고 지정  
#웨이트와 바이어스 보고싶으면 그것들은 feed_dict에서 나오는값으로 run 에 추가 시켜줘야한다
# 실질적으로 보이는것은 sess.run에서 넣은것들만 출력 할 수 있는것이다 
    
    if step %20 ==0: #20번 돌릴때마다 한번씩 출력 된다 step을 20으로 나눌때 나머지가 0 이면 print
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val,W_val, b_val)
#step %20 ==0 -> 왼쪽 변수에서 오른쪽값을 나눈후 나머지를 반환한다
print('d')