#실습 
#아웃라이어 확인 

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_wine 

datasets = pd.read_csv('D:\Study\_data\winequality-white.csv',
                        index_col=None, header=0, sep=';')


import matplotlib.pyplot as plt 
#datasets의 바 그래프를 그리시오 
#y데이터의 라벨당 갯수를 bar그래프로 그리시오 
count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

plt.bar(count_data.index, count_data)
plt.show()

# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5



