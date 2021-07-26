from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터 
docs = ['너무 재밌어요','참 최고예요','참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다', '한번 더 보고싶네요','글쎄요',
        '별로예요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밌네요','청순이가 잘 생기긴 했어요']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
#print(token.word_index)

x = token.texts_to_sequences(docs)
#print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13], 
# [14], [15], [16, 17], [18, 19], [20], [2, 21], [1, 22], 
# [23, 3, 24, 25]]

#리스트는크기가 달라도 되나 넘파이는 크기가 다르면 안된다
#data가 크기가 다를때 가장 큰값에 맞춰서 작은값에 0을 붙이겠다는 말
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) #pre 말고 post도 있고 pre는 앞에 0 붙여주는것 

print(pad_x)
print(pad_x.shape) #(13, 5)

word_size = len(token.word_index)
print(word_size) #25 라벨이 25개 
print(np.unique(pad_x)) #총 26개 
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]
#원핫인코딩 하면 어덯게 바뀌지요? (13,5) -> 라벨의 갯수만큼 바뀐다 
#(13,5) -> (13,5,25) 25-> 총라벨의 갯수 
#원핫인코딩을 할 경우 데이터가 너무 커진다 
#예를들어 사전을 하라경우 (13,5,10000000) -> 6500만개 데이터 너무큼 
#00001이면 필요한 데이터는 1이고 0000은 필요 없는 데이터이다 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense,LSTM,Conv1D
#자연어 처리에서 많이 사용하는 Embedding(벡터화를 시켜줘서 원핫인코딩이 필요가 없다)

#2. 모델 

model = Sequential()
model.add(Embedding(input_dim=26, output_dim=11, input_length=5))
#input_dim=라벨의 갯수(25까지 있으니 0부터니깐 총 26개) (단어사전의 갯수 -옥스포드 사전이 100개이면 101), 
# output_dim=11 (DNN = unit, CNN = filter와 같고 아웃풋 노드의 갯수 이다
#  유일하게 embedding만 아웃풋 노드가 가운데 있다 다른건 앞에 있다 ) 출력 노드의 갯수
# input_length=5 (문장의 길이 -최대 길이max_len의길이 )
#model.add(Embedding(25,77)) #input_dim, out_dim만 들어가고 input_length는 None으로 된다 그리고 최대 길이를 자동으로 인식하겠다는말 

model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#평가 예측 
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)

#acc :  0.7538461685180664