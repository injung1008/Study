import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#ImageDataGenerator = > data 수치화, 증폭하는 기능이 있다 
#horizontal_flip - 수평이동
#width_shift_range - 이미지 증폭시키는것과 관련
# #fill_mode='nearest' 근접에 이미지가 있을때 이미지와 매칭을 시켜가지고
#  사진을 옮겼을때 공백이 생기면안되니 공백부분을 비슷한것과 채운다는것 
#이런 기능을 가진것을 정의한것이다 
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

#있는 그대로의 사진을 가지고 검증을 하는게 목적인데 이걸 증폭시키면 데이터를 건든것이 되니깐 
#테스트데이터는 증폭하지 않는다 (rescale=1./255) 스케일은 훈련과 동일해야한다 )
test_datagen = ImageDataGenerator(rescale=1./255)

#train_datagen.flow_from_directory 을 통과하면서 x,y가 생성된다
xy_train = train_datagen.flow_from_directory(
    'D:\\Study\\_data\\brain\\train',
    target_size=(150,150), #이미지는 다 사이즈가 틀리기때문에 일률적으로 150,150으로 고정
    batch_size=5,
    class_mode='binary', # data를 라벨링하듯이 파일을 2개로 놓은 다음 하나는 0 하나는 1로 사진으로 두군집을 분류한다
    shuffle=False)

xy_test = test_datagen.flow_from_directory(
        'D:\\Study\\_data\\brain\\train',
        target_size=(150,150),
        batch_size=5,
        class_mode='binary'
)

#^^ 위에 꺼만 실행 시켰을때 Found 160 images belonging to 2 classes.(160장 사진) 

#print(xy_train) #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000292486E97F0>
# print(xy_train[0])
# print(xy_train[0][0]) 
#^^ [x][y]를 얘기해주고 뒤에가 [1]로 하면 y값 뽑아줌 뒤에 [0]이면 x값 뽑아줌
# print(xy_train[0][1]) #y [1. 1. 0. 0. 0.]
#print(xy_train[0][2]) # 없음
#! print(xy_train[0][0].shape, xy_train[0][1].shape)  
#! (5, 150, 150, 3) (5,) /  5 = 배치 (xy_train[0] 에 사진이 5장 들어있음)
# 예를들어 ad = 0 , nomal =1 이런식으로 찍힌다 
#총 160/5 = 32 이니깐 [0]-[1]-[2] ...[31] 까지 총 x 갯수가 32개라는말 
print(xy_train[31][1]) #y값이 나온다

# print(type(xy_train)) #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>       
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

print(type(xy_test[0][1])) #<class 'numpy.ndarray'>
