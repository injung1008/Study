import numpy as np 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout 

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

#2. 모델 
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='ouput')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model 

def create_hyperparameter():
    batches = [1000, 2000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop": dropout}

hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [10, 20, 30, 40, 50], 'optimizer': ['rmsprop', 'adam', 'adadelta'], 'drop': [0.1, 0.2, 0.3]}

# model2 = build_model() 

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # 텐서플로우 모델을 사이킥런 모델로 인식하게 만들어주는것 
model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=3,validation_split=0.2)
#keras모델을 kerasclassfier에 랩핑해주면 사이킥런에 사용할 수 있다 
#KerasClassifier = epochs 사용 가능 validation_split 사용가능 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
model = RandomizedSearchCV(model2, hyperparameters, cv=5)
#RandomizedSearchCV 기존에는 사이칵런 머신러닝만 넣어줫고 텐서플로우모델을 넣은적이 없다
#텐서플로우를 사이킥런에서 사용하고싶으면 어케해야할까? 
#텐서플로우 형식을 사이킥런 형식으로 바꿔주면 된다 
#텐서플로우 모델을 사이킥런으로 감싸주면 된다 -> 랩핑 하면된다 
#cv=5 kfold값을 그냥 넣어주면 알아서 kfold와 동일한 효과를 준다 

model.fit(x_train, y_train, verbose=1)#, validation_split=0.2) # ,epochs=3 

#KerasClassifier 보다 fit epochs가 우선 순위 
#1280/1280 [==============================] - 4s 3ms/step - loss: 2.2220 - acc: 0.2137 - val_loss: 2.1122 - val_acc: 0.4776

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)

# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 1000}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000024A022F4040>
# 0.9588333249092102
# 10/10 [==============================] - 0s 3ms/step - loss: 0.1079 - acc: 0.9663
# 최종 스코어 :  0.9663000106811523