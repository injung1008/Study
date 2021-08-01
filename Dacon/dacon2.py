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

from sklearn.model_selection import train_test_split

x, x_val, y, y_val = train_test_split(x, y, 
        train_size = 0.8, shuffle=True, random_state=9)



#print(type(y)) #<class 'numpy.ndarray'>
# print(len(x))
# print("최대길이 : ", max(list(map(len, x)))) 
# print(x)
#print(test_test)

tfidf = TfidfVectorizer(analyzer='char', sublinear_tf=True, ngram_range=(1, 2), max_features=45000, binary=False)

tfidf.fit(x)

# print(sorted(tfidf.vocabulary_.items()))

x = tfidf.transform(x).astype('float32')
x_val = tfidf.transform(x_val).astype('float32')
test_test  = tfidf.transform(test_test).astype('float32')
#print(x[0])




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

#! print(x.shape) #(45654, 76529)
#! print(type(x)) # <class 'scipy.sparse.csr.csr_matrix'>

model = Sequential()
# model.add(Embedding(input_dim=76500, output_dim=35, input_length=13))
# model.add(LSTM(152))
# model.add(Dropout(0.5))
model.add(Dense(200, input_dim=45000, activation='relu'))
model.add(Dropout(0.8))
#model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
#                     restore_best_weights=True)

hist = model.fit(x, y, epochs=10,batch_size=30, validation_data=(x_val,y_val))

#평가 예측 
p_acc = model.evaluate(x, y)
print("p_acc : ", p_acc)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss )
print('val loss : ', val_loss)


#print('acc 전체 : ', acc)

print('acc : ', acc[-1])
print('val acc : ', val_acc[-1])



temp = model.predict(test_test)

temp = tf.argmax(temp, axis=1)

temp = pd.DataFrame(temp)

temp.rename(columns={0:'topic_idx'}, inplace=True)

temp['index'] = np.array(range(45654,45654+9131))
temp = temp.set_index('index')
print(temp)
temp.to_csv('D:\Study\Dacon\_save\submission5.csv')