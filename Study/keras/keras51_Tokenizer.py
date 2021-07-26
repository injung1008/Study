from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)