import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense, Input,LSTM 
import tensorflow as tf
from datetime import datetime
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
df_sam.set_index('일자', inplace=True)
df_sk.set_index('일자', inplace=True)

df_sam = df_sam.sort_index(ascending=True)
df_sk = df_sk.sort_index(ascending=True)

#print(df_sam)

# print(df_sam)
# print(df_sk)
df_sam = pd.DataFrame(df_sam, columns=['시가','고가','저가','거래량','종가'])
df_sk = pd.DataFrame(df_sk, columns=['시가','고가','저가','거래량','종가'])

# print(df_sam) #[2601 rows x 5 columns]
# print(df_sk) #[2601 rows x 5 columns]

#samsung = df_sam.to_numpy()
sk = df_sk.to_numpy()


#학습데이터 생성하기size= 내가 얼마동안의 주가 데이터에 기반하여
#다음날 종가를 예측할 것인가를 정하는 파라미터이다. 즉 내가 과거 20일을 기반으로 내일데이터를
#예측한다고 가정했을때는 size =20이된다 

#test_size = 200은 학습은 과거부터  200일 이전의 데이터를 학습하게 되고, test를 위해서 이후200일의
#데이터로 모델이 주가를 예측하도록 한다음, 실제 데이터와 오차가 얼마나 있는지 확인 해본다 
#! train 만들기
test_size = 200
df_sam_train = df_sam[:-test_size] # 데이터 프레임은 행을 어디서 뽑을건지 [:]0-2400까지만 뽑는것 
df_sam_test = df_sam[-test_size:] #2400-2600 까지 뽑기 (현재데이터 기준 )


df_sk_train = df_sk[:-test_size] 
df_sk_test = df_sk[-test_size:] 




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
scale_cols = ['종가','고가','저가','거래량']

# df_scaled = scaler.fit_transform(df_sam[scale_cols]) #삼성값 전처리 훈련하기
# df_scaled = pd.DataFrame(df_scaled) #삼성 데이터프레임화 시키기
# df_scaled_sk = scaler.fit_transform(df_sk[scale_cols])
# df_scaled_sk = pd.DataFrame(df_scaled_sk) #sk 데이터프레임화 시키기
#삼성
scaler.fit(df_sam_train[scale_cols])
df_scaled_train = scaler.transform(df_sam_train[scale_cols])
df_scaled_train = pd.DataFrame(df_scaled_train)#삼성 데이터프레임화 시키기
df_scaled_test = scaler.transform(df_sam_test[scale_cols])
df_scaled_test = pd.DataFrame(df_scaled_test)#삼성 데이터프레임화 시키기

#sk

scaler2.fit(df_sk_train[scale_cols])
df_scaled_sk_train = scaler2.transform(df_sk_train[scale_cols])
df_scaled_sk_train = pd.DataFrame(df_scaled_sk_train)#삼성 데이터프레임화 시키기
df_scaled_sk_test = scaler2.transform(df_sk_test[scale_cols])
df_scaled_sk_test = pd.DataFrame(df_scaled_sk_test)#삼성 데이터프레임화 시키기

df_scaled_train.columns = ['종가','고가','저가','거래량'] #칼럼명 넣어주기 
df_scaled_test.columns = ['종가','고가','저가','거래량'] #칼럼명 넣어주기 
df_scaled_sk_train.columns = ['종가','고가','저가','거래량'] #칼럼명 넣어주기
df_scaled_sk_test.columns = ['종가','고가','저가','거래량'] #칼럼명 넣어주기

#삼성 시가 
df_sam_train_label = df_sam_train['시가'].reset_index()
df_sam_train_label = df_sam_train_label['시가']
df_sam_test_label = df_sam_test['시가'].reset_index()
df_sam_test_label = df_sam_test_label['시가']
#sk 시가
df_sk_train_label = df_sk_train['시가'].reset_index()
df_sk_train_label = df_sk_train_label['시가']
df_sk_test_label = df_sk_test['시가'].reset_index()
df_sk_test_label = df_sk_test_label['시가']
#print(df_sk_test_label)

# #다시 결합해주기
train= df_scaled_train.join(df_sam_train_label)
test= df_scaled_test.join(df_sam_test_label)
#sk 결합
train_sk = df_scaled_sk_train.join(df_sk_train_label)
test_sk = df_scaled_sk_test.join(df_sk_test_label)

#^^#######################################데이터 전처리구간 ###############################################

#dataset을 만들어주는 함수

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_lsit = []
    for i in range(len(data)- window_size -1):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        # i가 0번째라면 0-19번째까지 뽑아옴
        # 시가,고가,저가,거래량데이터를 20개뽑기
        # 그 다음날 예측이면 20번째 종가를 뽑아옴 2일뒤이면 +1해서 2일뒤꺼 뽑아옴 
        label_lsit.append(np.array(label.iloc[i+window_size+1]))    
    return np.array(feature_list), np.array(label_lsit)



# #^^
feature_cols = ['종가','고가','저가','거래량']
label_cols = ['시가']
#2400개의 데이터 
#split 돌리기 위해서 데이터 피쳐와 종가로 나눠주기 

#삼성
train_feature = train[feature_cols]
train_label = train[label_cols]
#test(200일 최신의것)
test_feature = test[feature_cols]
test_label = test[label_cols]

#sk
train_feature_sk = train_sk[feature_cols]
train_label_sk = train_sk[label_cols]
#test(200일 최신의것)
test_feature_sk = test_sk[feature_cols]
test_label_sk = test_sk[label_cols]

# print(test_label_sk)




#위에서 각각 나온 리스트 들을 넣어준다 
x_sam_train,y_sam_train = make_dataset(train_feature, train_label,20)
x_sk_train,y_sk_train = make_dataset(train_feature_sk, train_label,20)
#print(x_sam_train.shape,x_sk_train.shape) #(2379, 20, 4) (2379, 20, 4)
#! data로 2400개 트레인세트가 들어간다 (사이즈만큼 전체데이터가 줄고(중복때문에))
#!'종가' 가 오늘과 내일의 [] 데이터셋은 사용하지 않아도된다 왜냐면 어제의 데이터가 내일을 예측해주기때문


# # #! 실제로 테스트 해볼 최근 200일의 데이터 
x_sam_test, y_sam_test = make_dataset(test_feature, test_label,20)
x_sk_test, y_sk_test = make_dataset(test_feature_sk, test_label_sk,20)
# print(x_sk_test.shape,x_sk_train.shape)#(180, 20, 4) (2379, 20, 4)
# print(x_sam_test.shape,x_sam_train.shape)#(180, 20, 4) (2379, 20, 4)
# print(y_sam_test)

from keras.models import Sequential, Model
from keras.layers import Dense,Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten



#!model = load_model('D:\Study\Samsung\_save\ModelCheckPointsamsung_siga0723_1159_.0083-11962648.0000.hdf5') # [77052.05 ]]
#
#!model = load_model('D:\Study\Samsung\_save\ModelCheckPointsamsung_siga0723_1426_.0099-11302288.0000.hdf5') # [80701.25 ]]
#model = load_model('D:\Study\Samsung\_save\ModelCheckPointsamsung_siga0723_1426_.0100-12986655.0000.hdf5') # [77615.75 ]]
#^^ model = load_model('D:\Study\Samsung\_save\ModelCheckPointsamsung_siga0723_1426_.0098-11917253.0000.hdf5') # [82864.95 ]]
model = load_model('D:\Study\Samsung\_save\ModelCheckPointsamsung_siga0723_1426_.0086-11593410.0000.hdf5')
loss = model.evaluate([x_sam_test, x_sk_test],y_sam_test)

print('loss : ', loss )

result = model.predict([x_sam_test,x_sk_test])
print('삼성 : ', result)

#tensorboard --logdir=./logs/fit/