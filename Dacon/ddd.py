import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.feature_extraction.text import TfidfVectorizer



train = pd.read_csv('D:\Study\_data\형준\\train.csv', sep=',',
                         header=0)

train_y = pd.read_csv('D:\Study\_data\\train_data.csv', sep=',',
                         header=0)

test_test = pd.read_csv('D:\Study\_data\형준\\predict.csv', sep=',',
                        header=0)

train = train.fillna('c')
test_test = test_test.fillna('a')
train_y = train_y["topic_idx"]
# print(train.shape) #(45654, 1)
# print(train)
# print(train_y)
# print(train_y.shape) #(45654,)
# print(test_test.shape)
# print(test_test)





x = train['0'].tolist()
# print(x)
test_test =test_test["0"].tolist()
y = np.array(train_y)
# print(y)
from sklearn.model_selection import train_test_split

x, x_val, y, y_val = train_test_split(x, y, 
        train_size = 0.8, shuffle=True, random_state=9)



#print(type(y)) #<class 'numpy.ndarray'>
# print(len(x))
# print("최대길이 : ", max(list(map(len, x)))) 
# print(x)
#print(test_test)

# tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=24555, binary=False)

# tfidf.fit(x)

# # print(sorted(tfidf.vocabulary_.items()))

# x = tfidf.transform(x).astype('float32')
# x_val = tfidf.transform(x_val).astype('float32')
# test_test  = tfidf.transform(test_test).astype('float32')
# # print(x[0])


from tensorflow.keras.preprocessing.text import Tokenizer
token = Tokenizer()
token.fit_on_texts(x)
# print(token.word_index)
x = token.texts_to_sequences(x)
# print(np.max(x)) #22411
x_val = token.texts_to_sequences(x_val)
test_test = token.texts_to_sequences(test_test)

# print("뉴스 기사의 최대길이 : ", max(list(map(len, x)))) #15

from tensorflow.keras.preprocessing.sequence import pad_sequences


x = pad_sequences(x, maxlen=15, padding='pre')
x_val = pad_sequences(x_val, maxlen=15, padding='pre')
test_test = pad_sequences(test_test, maxlen=15, padding='pre')
# print(test_test)
# print(x.shape, y.shape) #(36523, 15) (36523,)
print(x[0]) #[   0    0    0    0    0  952   57  109   58  118  471 3185  110  529 1676]
# print(type(x_train), type(x_train[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>


# from tensorflow.keras.utils import to_categorical
# #print(np.unique(y_train)) #총 7개

# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout


model = Sequential()
model.add(Embedding(input_dim=36523, output_dim=35, input_length=15))
model.add(LSTM(152))
model.add(Dropout(0.8))
# model.add(Dense(36, activation='relu'))
# model.add(Dense(16))
model.add(Dense(7, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
#                     restore_best_weights=True)

model.fit(x, y, epochs=80, batch_size=100,  validation_data=(x_val,y_val))#, callbacks=[es])

# #평가 예측 
acc = model.evaluate(x_val, y_val)
print("acc : ", acc)

temp = model.predict(test_test)

temp = tf.argmax(temp, axis=1)
print('temp: ',temp)
temp = pd.DataFrame(temp)
print('y : ', y)
# temp.rename(columns={0:'topic_idx'}, inplace=True)

# temp['index'] = np.array(range(45654,45654+9131))
# temp = temp.set_index('index')
# print(temp)
# temp.to_csv('D:\Study\Dacon\_save\submission.csv')



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

# # print(x.shape) #(36523, 24555)
# # print(type(x)) # <class 'scipy.sparse.csr.csr_matrix'>

# model = Sequential()
# # model.add(Embedding(input_dim=76500, output_dim=35, input_length=13))
# # model.add(LSTM(152))
# # model.add(Dropout(0.5))
# model.add(Dense(100, input_dim=24555, activation='relu'))
# model.add(Dropout(0.7))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))
# model.summary()

# # #3. 컴파일, 훈련 

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # from keras.callbacks import EarlyStopping
# # es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
# #                     restore_best_weights=True)

# hist = model.fit(x, y, epochs=15,batch_size=20, validation_data=(x_val,y_val))

# #평가 예측 
# p_acc = model.evaluate(x, y)
# print("p_acc : ", p_acc)

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss )
# print('val loss : ', val_loss)


# #print('acc 전체 : ', acc)

# print('acc : ', acc[-1])
# print('val acc : ', val_acc[-1])



# temp = model.predict(test_test)

# temp = tf.argmax(temp, axis=1)

# temp = pd.DataFrame(temp)

# temp.rename(columns={0:'topic_idx'}, inplace=True)

# temp['index'] = np.array(range(45654,45654+9131))
# temp = temp.set_index('index')
# print(temp)
# temp.to_csv('D:\Study\Dcon_save\_save\submission.csv')
