
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
train=pd.read_csv('D:\Study\_data\open\open\\train.csv')
test=pd.read_csv('D:\Study\_data\open\open\\test.csv')
sample_submission=pd.read_csv('D:\Study\_data\open\open\sample_submission.csv')

#심각한 불균형 데이터임을 알 수 있습니다.
train.label.value_counts(sort=False)/len(train)

#해당 baseline 에서는 과제명 columns만 활용했습니다.
#다채로운 변수 활용법으로 성능을 높여주세요!
train=train[['과제명','label']]
test=test[['과제명']]

#1. re.sub 한글 및 공백을 제외한 문자 제거
#2. okt 객체를 활용해 형태소 단위로 나눔
#3. remove_stopwords로 불용어 제거 
def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
    word_text=okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review
stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
okt=Okt()
clean_train_text=[]
clean_test_text=[]
#시간이 많이 걸립니다.
for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

from sklearn.feature_extraction.text import CountVectorizer

#tokenizer 인자에는 list를 받아서 그대로 내보내는 함수를 넣어줍니다. 또한 소문자화를 하지 않도록 설정해야 에러가 나지 않습니다.
vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(clean_train_text)
test_features=vectorizer.transform(clean_test_text)
#test데이터에 fit_transform을 할 경우 data leakage에 해당합니다

#훈련 데이터 셋과 검증 데이터 셋으로 분리
TEST_SIZE=0.2
RANDOM_SEED=42

train_x, eval_x, train_y, eval_y=train_test_split(train_features, train['label'], test_size=TEST_SIZE, random_state=RANDOM_SEED)
#랜덤포레스트로 모델링
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split, KFold,  GridSearchCV

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100,200]},
    {'learning_rate' : [0.1,0.001,0.05]},
    {'max_depth' : [4,5,6]},
    # {'colsample_bytree' : [0.6,0.9,1]},
    # {'colsample_bylevel' : [0.6,0.7,0.9]},
    # {'min_samples_leaf' : [3, 5, 7, 10]},
    # {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

#!2. model 구성 
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

model.fit(train_x, train_y)

#모델 검증
result = model.score(eval_x, eval_y)
print('result : ',result)

#4. 평가, 예측 
print("최적의 매개변수 : ", model.best_estimator_) # -> train에 대한 평가값 
#best_estimator_ 가장 좋은 평가가 무엇인가? 
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("best_score_ : ", model.best_score_)

model.predict(test_features)
sample_submission['label']=model.predict(test_features)
sample_submission.to_csv('_baseline.csv', index=False)


