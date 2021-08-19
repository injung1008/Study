#실습 다차원의 oulier가 출력되도록 함수 수정 

import numpy as np   
aaa = np.array([[1,2,10000,3,4,6,7,8,90,100, 5000],
            [1000,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])

aaa = aaa.transpose()
# print(aaa.shape)

def outliers(data_out):
    for i in range(1,3):
        print(i)
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i-1:i],[25,50,75])
        print("1사분위 : ", quartile_1)
        print("q2 :", q2)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print('i',iqr)
        print('l',lower_bound)
        print('u',upper_bound)
        print(data_out[:,i-1:i]<lower_bound)
        print(data_out[:,i-1:i]>upper_bound) 
        print(np.where((data_out[:,i-1:i]>upper_bound) | (data_out[:,i-1:i]<lower_bound)))
    # return np.where((data_out[:,i-1:i]>upper_bound) | (data_out[:,i-1:i]<lower_bound))
       


# outliers_loc = outliers(aaa[:,:1])
# outliers_loc = outliers(aaa[:,:-1])

outliers_loc = outliers(aaa)
print('이상치의 위치 : ', outliers_loc)


# 시각화 
# 위 데이터를 boxplot으로 그리시오 
import matplotlib.pyplot as plt 
plt.boxplot(aaa)
plt.show()