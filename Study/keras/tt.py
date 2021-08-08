#훈련데이터를 기존 데이터에서 20% 만큼 증폭 
#성과비교
#save_dir도 temp에 넣은후 삭제할것 

from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=False,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    rotation_range=5,
                    zoom_range=0.1,
                    shear_range=0.5,
                    fill_mode='nearest'
                    )


augment_size = 10000 #배치사이즈
# print(x_train.shape) #(50000, 28, 28)
randidx = np.random.randint(x_train.shape[0], size=augment_size)
                              #가져올놈         사이즈만큼 가져올놈에서 데려오겠다 


