import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_csv('./_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)
# ./ : 현재폴더 ../: 상위폴더

# print(df)
# print(df.shape) #(4898, 12) 


# print(datasets.describe())
#다중분류\#모델링하고
#0.8이상 완성

# x = datasets.data
# y = datasets.target

#1. 판다스를 넘파이로 바꾼다
#2. x와 y를 분리
#3. y의 라벨을 확인 np.unique

#넘파이로 변경전에 미리 y,x를 분리해준다 넘파이로 변경했을때 data,target구분이 안되기
#때문에 미리 구분을 해주고 진행 하는것이 좋다 


datasets = df.to_numpy()

x = datasets[:,0:11]
y = datasets[:,[-1]]


# print(np.unique(y)) #7개 [3 4 5 6 7 8 9]




# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(y)
y = onehot_encoder.transform(y).toarray(y)
print(y)




# x_train, x_test, y_train, y_test =train_test_split(x,y,
#     train_size = 0.7, shuffle = True, random_state=66)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
scaler = PowerTransformer()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = Sequential()
# model.add(Dense(1, input_dim=11)) 
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax')) #y lable 3ea

# #훈련
# model.compile(loss='categorical_crossentropy',optimizer='adam',
#                     metrics=['accuracy'])

# # from tensorflow.keras.callbacks import EarlyStopping

# # es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)

# model.fit(x_train, y_train, epochs=100, verbose=1,
# batch_size=1, validation_split=0.2)

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('metrix : ', loss[1] )

