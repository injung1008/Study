import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.feature_extraction.text import TfidfVectorizer

test = pd.read_csv('D:\\Study\\_data\\test_data.csv',sep=',',
                          index_col='index',header=0)


topic = pd.read_csv('D:\\Study\\_data\\topic_dict (1).csv',
                        index_col=None, header=0)


train = pd.read_csv('_data\\train_data.csv', sep=',',
                        index_col='index', header=0)
def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean
    


train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

x = train["cleaned_title"].tolist()
test_test =test["cleaned_title"].tolist()
y = np.array(train.topic_idx)
#print(type(y)) #<class 'numpy.ndarray'>

#print(test_test)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=76529, binary=False)

tfidf.fit(x)

x = tfidf.transform(x).astype('float32')
test_test  = tfidf.transform(test_test).astype('float32')






#! 넘파이 변환 
# x = np.array(train["cleaned_title"])
# y = np.array(train['topic_idx'])


# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
# train_size = 0.95, shuffle=True, random_state=66)

# #print(x_train.shape,y_train.shape) #(43371,) (43371,)
# # print(x)
# #print(x_test.shape) #(2283,)


# from tensorflow.keras.preprocessing.text import Tokenizer
# token = Tokenizer()
# token.fit_on_texts(x_train)
# #print(token.word_index)
# x_train = token.texts_to_sequences(x_train)
# #print(np.max(x_train))
# x_test = token.texts_to_sequences(x_test)
# test_test = token.texts_to_sequences(test_test)


#print("뉴스 기사의 최대길이 : ", max(list(map(len, x)))) #13

# from tensorflow.keras.preprocessing.sequence import pad_sequences


# x_train = pad_sequences(x_train, maxlen=13, padding='pre')
# x_test = pad_sequences(x_test, maxlen=13, padding='pre')
# test_test = pad_sequences(test_test, maxlen=13, padding='pre')
# #print(test_test)
# # print(x_train.shape, x_test.shape) #(36523, 13) (9131, 13)
# # print(x_train[0])
# # print(type(x_train), type(x_train[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>


# from tensorflow.keras.utils import to_categorical
# #print(np.unique(y_train)) #총 7개[0,1,2,3,4,5,6]

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

#! print(x.shape) #(45654, 76529)
#! print(type(x)) # <class 'scipy.sparse.csr.csr_matrix'>

model = Sequential()
# model.add(Embedding(input_dim=76529, output_dim=35, input_length=13))
# model.add(LSTM(152))
# model.add(Dropout(0.5))
model.add(Dense(128, input_dim=76529, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(7, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
#                     restore_best_weights=True)

model.fit(x, y, epochs=6, batch_size=100)

#평가 예측 
acc = model.evaluate(x, y)[1]
print("acc : ", acc)

temp = model.predict(test_test)

temp = tf.argmax(temp, axis=1)

temp = pd.DataFrame(temp)

temp.rename(columns={0:'topic_idx'}, inplace=True)

temp['index'] = np.array(range(45654,45654+9131))
temp = temp.set_index('index')
print(temp)
temp.to_csv('D:\Study\Dacon\_save\submission.csv')