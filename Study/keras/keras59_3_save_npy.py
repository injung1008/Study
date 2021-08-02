# <배치사이즈 별로 짤려있는것을 합치기> 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#! 1. data

#ImageDataGenerator = > data 수치화, 증폭하는 기능이 있다 
#horizontal_flip - 수평이동
#width_shift_range - 이미지 증폭시키는것과 관련
# #fill_mode='nearest' 근접에 이미지가 있을때 이미지와 매칭을 시켜가지고
#  사진을 옮겼을때 공백이 생기면안되니 공백부분을 비슷한것과 채운다는것 
#이런 기능을 가진것을 정의한것이다 
train_datagen = ImageDataGenerator(rescale=1./255,
                    # horizontal_flip=True,
                    # vertical_flip=True,
                    # width_shift_range=0.1,
                    # height_shift_range=0.1,
                    # rotation_range=5,
                    # zoom_range=1.2,
                    # shear_range=0.7,
                    # fill_mode='nearest'
                    )

#있는 그대로의 사진을 가지고 검증을 하는게 목적인데 이걸 증폭시키면 데이터를 건든것이 되니깐 
#테스트데이터는 증폭하지 않는다 (rescale=1./255) 스케일은 훈련과 동일해야한다 )
test_datagen = ImageDataGenerator(rescale=1./255)

#train_datagen.flow_from_directory 을 통과하면서 x,y가 생성된다
xy_train = train_datagen.flow_from_directory(
    'D:\\Study\\_data\\brain\\train',
    target_size=(150,150), #이미지는 다 사이즈가 틀리기때문에 일률적으로 150,150으로 고정
    batch_size=200,
    class_mode='binary', # data를 라벨링하듯이 파일을 2개로 놓은 다음 하나는 0 하나는 1로 사진으로 두군집을 분류한다
    shuffle=True)

xy_test = test_datagen.flow_from_directory(
        'D:\\Study\\_data\\brain\\test',
        target_size=(150,150),
        batch_size=200,
        class_mode='binary'
)
np.save('D:\Study\_npy\k59_3_train_x.npy', arr=xy_train[0][0])
np.save('D:\Study\_npy\k59_3_train_y.npy', arr=xy_train[0][1])
np.save('D:\Study\_npy\k59_3_test_x.npy', arr=xy_test[0][0])
np.save('D:\Study\_npy\k59_3_test_y.npy', arr=xy_test[0][1])


#^^  배치사이즈를 늘려서 모든 데이터가 한번에 다 들어 갈수 있게 해준다 

print(xy_train[0][0].shape, xy_train[0][1].shape) #(160, 150, 150, 3) 
# print(xy_test[0][0].shape, xy_test[0][1]) #(120, 150, 150, 3) 



# #! 2. model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(32,(2,2), input_shape=(150,150,3)))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# #x y가 붙어있는경우 fit_generator사용하면 fit과 동일한 결과 나옴
# hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32, # steps_per_epoch=32 한 에포당 돌아가는 훈련양 ? 160/5 = 32
#                     validation_data=xy_test)
#                    # validation_steps=4) 
#                     #validation_steps=4 이런 파라미터가 있다 

# import matplotlib.pyplot as plt

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']


# # plt.imshow(loss, 'gray')
# # plt.show()

# #위에것으로 시각화 할것 

# print('acc : ', acc[-1])
# print('val acc : ', val_acc[:-1])