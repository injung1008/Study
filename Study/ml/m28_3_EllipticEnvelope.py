#이상치 처리
#1. 삭제
#2. nan처리후 -> 보간 // linear
#3. ............(결측치 처리 방법과 유사)
#4. scaler -> Rubsorscaler, QuantileTransfomer ... 등등등 
#5. 모델링 : tree 계열 ... DT, RF, XG, LGBM...  결측치에서 자유롭다 

import numpy as np 
aaa = np.array([[1,   2,  10000, 3,4,6,7,8,90,100,5000],
                [1000, 2000, 3,  4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])

aaa = aaa.transpose()
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope 

outliers = EllipticEnvelope(contamination=.3)
outliers.fit(aaa) 

results = outliers.predict(aaa)

print(results)#[ 1  1 -1  1  1  1  1  1  1  1 -1]
#[ 1  1 -1  1  1  1  1 -1 -1 -1 -1] 0.5
