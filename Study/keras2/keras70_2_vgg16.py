from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
vgg16.trainable=True #가중치를 동결한다 
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.trainable=False #이후에 추가된 모델에 대해서도 가중치 추가가 없다
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# vgg16 (Functional)           (None, 3, 3, 512)         14714688
# _________________________________________________________________
# flatten (Flatten)            (None, 4608)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                46090
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 14,760,789
# Trainable params: 46,101
# Non-trainable params: 14,714,688


print(len(model.weights)) 
#26(바이어스하나 레이어 하나 2개 * 13) -> 30 (트레이너블 파람은 0이 되고 26에서 레이어 2개를 늘려서 2*2 )
print(len(model.trainable_weights)) #0 -> 4
print(len(model.non_trainable_weights))