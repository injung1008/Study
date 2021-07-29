#실습 
#categorical_crossentropy & sigmoid 사용 


import numpy as np    
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터 불러오기 

train_datagen = ImageDataGenerator(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    rotation_range=5,
                    zoom_range=1.2,
                    shear_range=0.7,
                    fill_mode='nearest'
                    )

test_datagen = ImageDataGenerator(rescale=1./255)

#cat dog  1418 + 1912

xy_train = train_datagen.flow_from_directory(
    'D:\\Study\\_data\\cat_dog\\train',
    target_size=(150,150), #이미지는 다 사이즈가 틀리기때문에 일률적으로 150,150으로 고정
    batch_size=8200,
    class_mode='categorical', # data를 라벨링하듯이 파일을 2개로 놓은 다음 하나는 0 하나는 1로 사진으로 두군집을 분류한다
    shuffle=False)

xy_test = test_datagen.flow_from_directory(
        'D:\\Study\\_data\\cat_dog\\test',
        target_size=(150,150),
        batch_size=2300,
        class_mode='categorical'
)
#print(xy_train[0][0].shape, xy_train[0][1].shape) #(3309, 150, 150, 3) (3309,)
 
# print(xy_test[0][1])#(10, 150, 150, 3)

np.save('D:\Study\_npy\k59_cat_train_x.npy', arr=xy_train[0][0])
np.save('D:\Study\_npy\k59_cat_train_y.npy', arr=xy_train[0][1])
np.save('D:\Study\_npy\k59_cat_test_x.npy', arr=xy_test[0][0])
np.save('D:\Study\_npy\k59_cat_test_y.npy', arr=xy_test[0][1])
