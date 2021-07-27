from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.layers.recurrent import LSTM

#@ 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요', 
    '별로에요', '생각보다 지루해요', '연기가 어색해요', 
    '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
'''
{'참': 1, '너무': 2, '잘': 3, '재밋어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}
'''

x = token.texts_to_sequences(docs)
print(x)


from tensorflow.keras.preprocessing.sequence import pad_sequences 
# 리스트는 크기가 달라도 된다

pad_x = pad_sequences(x, padding='pre', maxlen=5) # post는 뒤에 추가
# maxlen을 길이보다 작게 설정하면 앞이 잘린다


print(pad_x)
print(pad_x.shape) #(13, 5)

print(np.unique(pad_x)) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

# 원핫인코딩을 하면 (13, 5) -> (13, 5, 27)
# 옥스포드 사전(13, 5, 1000000) -> 6500만개



word_size = len(token.word_index)
print(word_size)  # 27
 
#@ 2.모델
#. 유클리드란?
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding , LSTM , Bidirectional # Embedding 자연어 쪽에서 많이 씀

model = Sequential()  
                # 라벨의 개수  = 단어사전의 개수, output_dim = unit = filter = 아웃 풋 노드의 개수  ,input_length = 문장의 길이 

# 인풋은 13,5 단어수, 길이
model.add(Embedding(input_dim=28, output_dim=77, input_length=5)) # 원핫인코딩 + 백터화 /// 프리프로세스 개념이긴하나 인풋 단계에서 해결함

# model.add(Embedding(27, 77))도 가능 // 3차원으로 출력
# model.add(LSTM(32)) # 3차원의 대표인 lstm으로 받아줌
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation="sigmoid"))

'''
'''
exit()
model.summary()

#@ 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#@ 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)





