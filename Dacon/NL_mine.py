import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.feature_extraction.text import TfidfVectorizer

train=pd.read_csv('D:\Study\_data\open\open\\train.csv')
test=pd.read_csv('D:\Study\_data\open\open\\test.csv')
test_test=pd.read_csv('D:\Study\_data\open\open\sample_submission.csv')

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean
    



train['과제명'] = train['과제명'].apply(lambda x : clean_text(x))
test['과제명']  = test['과제명'].apply(lambda x : clean_text(x))
#

train=train[['과제명', '요약문_연구내용','label']]
test=test[['과제명', '요약문_연구내용']]
train['요약문_연구내용'].fillna('NAN', inplace=True)
test['요약문_연구내용'].fillna('NAN', inplace=True)


train['data']=train['과제명']+train['요약문_연구내용']
test['data']=test['과제명']+test['요약문_연구내용']


x = train["data"].tolist()
test_test =test["data"].tolist()
y = np.array(train.label)

# print(y[:5])
from sklearn.model_selection import train_test_split

x, x_val, y, y_val = train_test_split(x, y, 
        train_size = 0.9, shuffle=True, random_state=2)



# # print(type(y)) #<class 'numpy.ndarray'>
# # print(len(x))
# # print("최대길이 : ", max(list(map(len, x)))) 
# # print(x)
# # print(test_test)

tfidf = TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, ngram_range=(2, 6), max_features=45000, binary=False)

tfidf.fit(x)

# print(sorted(tfidf.vocabulary_.items()))

x = tfidf.transform(x).astype('float32')
x_val = tfidf.transform(x_val).astype('float32')
test_test  = tfidf.transform(test_test).astype('float32')
print(x[0])

from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(x,y)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

# print(x.shape) #(156873, 45000)
#! print(type(x)) # <class 'scipy.sparse.csr.csr_matrix'>

model = Sequential()
# model.add(Embedding(input_dim=76500, output_dim=35, input_length=13))
# model.add(LSTM(152))
# model.add(Dropout(0.5))
model.add(Dense(120, input_dim=45000, activation='relu'))
model.add(Dropout(0.8))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(14, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련 

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,
                    metrics=['acc'])
#지정하는건 컴파일이지만 실행하는건 fit이다 ->  일정한 값이 적용되지 않으면 러닝레이트를 줄이는것은 callbacks

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

##############################################################################                    
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/ModelCheckPoint'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
#filename = epoch값과 loss값이 파일명에 나올것이다 
modelpath = "".join([filepath, "dacon_", date_time, "_", filename])

#체크포인트가 갱신될때마다 파일이 생성이 된다 
#실질적으로 맨 마지막이 가장 높다
################################################################################3

cp = ModelCheckpoint(monitor = 'val_loss', mode='auto', batch_size = 8,verbose=1,
                        filepath = modelpath)    

# import time
# start_time = time.time()
es = EarlyStopping(monitor='val_acc', patience=2, mode='max', verbose=1)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, mode='auto', verbose=1, factor=0.05)
# #es는 조건이맞으면 끝나고, reduce_lr도 조건 맞으면 끝나지만, factor=0.5 로 해놓으면 감소가 없으면 러닝 레이트가 0.5 만큼 줄어든다.


hist = model.fit(x, y, epochs=1, verbose=1,validation_data=(x_val,y_val),
batch_size=512)#,callbacks=[cp])


# end_time = time.time() - start_time

#평가 예측 
p_acc = model.evaluate(x, y)
print("p_acc : ", p_acc)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val loss : ', val_loss[-1])


#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('val acc : ', val_acc)



temp = model.predict(test_test)

temp = tf.argmax(temp, axis=1)

temp = pd.DataFrame(temp)

temp.rename(columns={0:'topic_idx'}, inplace=True)

temp['index'] = np.array(range(174304,174304+43576))
temp = temp.set_index('index')
print(temp)
temp.to_csv('D:\Study\Dcon_save\_save\submission_NL.csv')