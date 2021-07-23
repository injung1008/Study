import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input,LSTM


df_sk = pd.read_csv('D:\Study\_data\SK주가 20210721.csv', 
                        index_col=None, header=0, encoding='cp949')


df_sam = pd.read_csv('D:\Study\_data\삼성전자 주가 20210721.csv',
                        index_col=None, header=0, encoding='cp949')

#print(df_sam.columns)

df_sam = pd.DataFrame(df_sam, columns=['일자','시가','고가','저가','거래량','종가'])
df_sk = pd.DataFrame(df_sk, columns=['일자','시가','고가','저가','거래량','종가'])

#print(df_sam.shape) # [3601 rows x 6 columns]
#rint(df_sk.shape) # [3600 rows x 6 columns]

df_sam = df_sam.loc[:2600]
df_sk = df_sk.loc[:2600]

# print(df_sam)
# print(df_sk)

#내림차순으로 바꿔주기 
df_sam = df_sam.sort_values(by=['일자'])
df_sk = df_sk.sort_values(by=['일자'])

# print(df_sam)
# print(df_sk)
df_sam = pd.DataFrame(df_sam, columns=['시가','고가','저가','거래량','종가'])
df_sk = pd.DataFrame(df_sk, columns=['시가','고가','저가','거래량','종가'])

# print(df_sam) #[2601 rows x 5 columns]
# print(df_sk) #[2601 rows x 5 columns]

samsung = df_sam.to_numpy()
sk = df_sk.to_numpy()

# print(df_sam)
# print(df_sk)
# print(samsung)
# print(sk)
def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

samsung = split_x(samsung,5)
sk = split_x(sk,5)


# print(samsung.shape) #(2597, 5, 5)
# print(sk.shape) #(2597, 5, 5)
sam_x = samsung[:,:4]
sam_y = samsung[:,4]
sk_x = sk[:,:4]
sk_y = sk[:,4]

# print(sam_x.shape) #(2597, 4, 5)
# print(sam_y.shape) #(2597, 5)
# print(sk_x.shape) #(2597, 4, 5)
# print(sk_y.shape) #(2597, 5)


sam_x_train,sam_x_test,sk_x_train,sk_x_test,sam_y_train, sam_y_test,sk_y_train, sk_y_test = train_test_split(sam_x,sk_x, 
                                                        sam_y,sk_y, train_size = 0.7, random_state=66)
# print(sam_x_train.shape) #(1817, 4, 5)
# print(sk_x_train.shape) #(1817, 4, 5)
# print(sam_x_test.shape) #(780, 4, 5)
# print(sam_x_test.shape) #(780, 4, 5)


sam_x_train = sam_x_train.reshape(1817, 20)
sk_x_train = sk_x_train.reshape(1817, 20)
sam_x_test = sam_x_test.reshape(780, 20)
sk_x_test = sk_x_test.reshape(780, 20)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(sam_x_train, sk_x_train)
sam_x_train = scaler.transform(sam_x_train)
sk_x_train = scaler.transform(sk_x_train)
sam_x_test = scaler.transform(sam_x_test)
sk_x_test = scaler.transform(sk_x_test)



#차원복귀
sam_x_train = sam_x_train.reshape(1817, 4,5)
sk_x_train = sk_x_train.reshape(1817, 4,5)
sam_x_test = sam_x_test.reshape(780, 4,5)
sk_x_test = sk_x_test.reshape(780, 4,5)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

#2-1 모델1 구성 
input1 = Input(shape=(4,5))
dense1 = LSTM(units=104, activation='relu')(input1) #히든 레이어 구성 
dense2 = Dense(104)(dense1)
dense3 = Dense(70)(dense2)
output1 = Dense(1)(dense3)

#2-2 모델2 구성 
input2 = Input(shape=(4,5))
dense11 = LSTM(units=130, activation='relu')(input2)
dense12 = Dense(120, activation='relu')(dense11)
dense13 = Dense(70, activation='relu')(dense12)
dense14 = Dense(20, activation='relu')(dense13)
output2 = Dense(1)(dense14)



from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)


#merge3에서 갈라져서 나옴
output21 = Dense(7,name='abc')(merge3)
last_output1 = Dense(1,name='abcd')(output21)

#sk y_train
output22 = Dense(8)(merge3)
last_output2 = Dense(1)(output22)


#concatenate, Concatenate 소문자와 대문자의 차이는 소문자 = 메소드, 대문자- 클래스
model = Model(inputs=[input1,input2] , outputs=[last_output1,last_output2])


model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit([sam_x_train,sk_x_train], [sam_y_train,sk_y_train], epochs=200, batch_size=8, verbose=1 )

loss = model.evaluate(sma_x_test, sam_y_test)

print('loss : ', loss )

result = model.predict([sam_y_test,sk_y_test])
#x_input = np.array([])
print('삼성 : ', result[0])

b