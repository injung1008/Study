import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
#1. 데이터 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.python.distribute.distribute_lib import Strategy

(x_train, y_train),(x_test, y_test) = mnist.load_data()


# #전처리



x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray() #(60000, 10)
y_test = encoder.transform(y_test).toarray() #(10000, 10)
print(y_test.shape)



#분산처리
#배치사이즈가 클 수록 좋다 
# strategy = tf.distribute.MirroredStrategy()
#gpu두개 같이 돌려서 사용할때 
# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
#     #tf.distribute.HierarchicalCopyAllReduce()
#     tf.distribute.ReductionToOneDevice())

# strategy = tf.distribute.MirroredStrategy(
#     #devices=['/gpu:0'] #0번 gpu만 돌리게 하는것 
#     #devices=['/gpu:1']
#     #devices=['/cpu', '/gpu:0']
#     # devices=['/cpu', '/gpu:0', '/gpu:1'], #구림 
#     # devices=['/gpu:0', '/gpu:1'] #구림

# )

#두개다 사용가능 
# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    # tf.distribute.experimental.CollectiveCommunication.AUTO
)



with strategy.scope():
    model = Sequential()
    model.add(Dense(512, input_dim=784))                                        
    model.add(Dense(251, activation='relu'))
    model.add(Dense(135, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=100, verbose=1, validation_split=0.025,
batch_size=300)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('metrix : ', loss[1] )



#acc로만 판단 
#metrix :  0.9714000225067139 , epochs=100
# metrix :  0.9793000221252441