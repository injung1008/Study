from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten 

# model = Sequential()
# model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1)))
# #^^ 가로가 28, 세로28, 1=흑백,3=칼라 -> 이미지 input_shape 데이터의 정보만 
# #^^ 만 명시하고 데이터의 수량은 명시하지 않음 
# #^^ kernel_size=(2,2) 가로2 세로2로 잘라서 작업을 하겠다는 의미 
# #^^ Conv2D(10, => 다음 레이어로 넘겨주는 노드의 수 
# model.add(Conv2D(20, (2,2))) #인풋사이즈가 위에서 명시 되어있으니 생략가능 (커널사이즈도 생략가능)
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32))
# model.add(Dense(1, activation='sigmoid')) #이진분류 
# model.summary()

#padding='same'  = 겨울에 패딩을 입으면 두꺼워 지지만 나의 본체가 두꺼워지는 건 아니다
#데이터에 가장자리에 
model = Sequential()                                              #(N,5,5,1)
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(5, 5, 1)))   #(N,4,4,10)- 패딩전 $ (N,5,5,10)-패딩후
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(30, (2,2)))                                      #(N,3,3,20)
model.add(Flatten())                                              #(N,180)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid')) # 왜 시그모이드를 사용할까? 
model.summary()