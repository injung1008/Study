# 다중분류 ( y가 0,1,2로 이루어져 있을때)
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150,4) (150,)
print(y) # y = 0,1,2
#0,1,2가 숫자의 값이 아닌 라벨이라고 생각하게 해야한다 
# 왜냐하면 2라는 것이 1의 두배의 값을 가진 개념이 아니고 그냥 이름이 2
#여자1 남자 2 라고 해서 남자가 여자의 2배의 가치를 가진게 아닌것과 같은의미
#데이터가 있는 위치에만 1을 넣어준다 (행렬에서 )
#전체 있는 데이터에서 해줘야지 바뀐 값으로 훈련데이터 시험데이터를 나누기떄문에 이전에 진행

# One Hot Encoding (150,) -> (150,3) #라벨수 3개 만큼 열이 늘어난다 
# 0 -> [1,0,0]
# 1 -> [0,1,0]
#2 -> [0,0,1]
#
# 예시 라벨이 4개일 때 [0, 1 , 2 , 1 ] -> 
#[[1,0,0]
#[0,1,0]
#[0,0,1]
#[0,1,0]] -> 4행 짜리가 -> 4행 3열  라벨의 종류만큼 칼럼(열)이 늘어난다 

from tensorflow.keras.utils import to_categorical #one hot 라이브러리
y = to_categorical(y) #위에서 벡터로 바꿔주는 과정을 얘가 처리해줌 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size = 0.7, shuffle = True, random_state=66,stratify=y)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(1, input_dim=4)) # x가 (150,4)여서 행무시 열우선으로 인풋 노드4개
model.add(Dense(100, activation='relu'))
model.add(Dense(123, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(3, activation='softmax')) #y가 (150,3) 이기 때문에 아웃풋 노드 3개 
#soft max도 나가는 값이 3개로 한정되는데 0.7,0.2/0.1이 나온다면 제일 큰ㄴ값이 
#0.7을 인저을 하고 얘를 1로 간주 하고 [1,0,0] 이라고 판단한다
#소프트 맥스는 분류된 확률의 합이 1이고 가장 큰 확률을 가진 친구가 1이 되고 그 위치를 기반으로 어떤 라벨인지 확인

model.compile(loss='categorical_crossentropy', optimizer='adam'
                        , metrics=['accuracy']) 

from tensorflow.keras.callbacks import EarlyStopping
#monitor를 기준으로 멈추겠다는 얘기 
# patience = loss값이 갱신되지 않는것을 5번까지는 참겠다, patience는 훈련수를 넘지 않는다
#mode = 'min' 로스 최저값이 5번을 갱신하면 멈추겠다 
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=50, verbose=1, 
            batch_size=1, validation_split=0.2 ,callbacks=[es])
# print(hist.history['loss'])
# print(hist.history['val_loss'])

print("=======평가예측==========")
loss = model.evaluate(x_test, y_test)
#이제 여기서 나오는 loss는 binary crossentropy이다 
#evaluate는 loss와 metrix도 리스트 형태로 같이 반환해준다 
print('loss : ', loss )
print('loss : ', loss[0] )
print('metrix : ', loss[1] )

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)




