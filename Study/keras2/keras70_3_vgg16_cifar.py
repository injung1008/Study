from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
vgg16.trainable=False #가중치를 동결한다 
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.summary()



# print(len(model.weights)) 
# #26(바이어스하나 레이어 하나 2개 * 13) -> 30 (트레이너블 파람은 0이 되고 26에서 레이어 2개를 늘려서 2*2 )
# print(len(model.trainable_weights)) #0 -> 4

#########################2번 파일에서 아래만 추가 ###################################3
import pandas as pd 
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable)for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Later Name', 'Layer Trainable'])

print(results)

#                                                                             Layer Type Later Name  Layer Trainable
# 0  <tensorflow.python.keras.engine.functional.Functional object at 0x000002AFE1571B50>  vgg16      False (trainable이 false인상태)
# 1  <tensorflow.python.keras.layers.core.Flatten object at 0x000002AF84463CA0>           flatten    True
# 2  <tensorflow.python.keras.layers.core.Dense object at 0x000002AF8445AC70>             dense      True
# 3  <tensorflow.python.keras.layers.core.Dense object at 0x000002AF8445AC10>             dense_1    True