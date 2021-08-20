from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False)#, input_shape=(32,32,3))
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, None, None, 3)]   0

model.trainable=False #가중치의 갱신이 없다 
# model.trainable=True #가중치의 갱신이 없다 

model.summary()

print(len(model.weights)) #26(trainable=True,false)
print(len(model.trainable_weights)) #26(trainable=True) 0(trainable=false)

# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688



# model = VGG16()
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# model = VGG19()
# model.summary()
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0

# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _____...____________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 #(Dense)                  (None, 4096)              102764544
# fc = fully connected_
# ________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================