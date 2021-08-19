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

#정규분포를 이용하여 데이터 분포에 타원을 그립니다. 타원에서 벗어날 수록 이상치이다
from sklearn.covariance import EllipticEnvelope 
#EllipticEnvelope를 이용하여 이상치를 검출하기 위한 객체생성 
#contamination의 비율을 기준으로 비율보다 낮을 값을 검출한다 
outliers = EllipticEnvelope(contamination=.3)
#EllipticEnvelope 객체를 생성한 데이터에 맞게 학습을한다 
outliers.fit(aaa) 

results = outliers.predict(aaa)

# outlier를 검출 합니다.
# +1 이면 boundary 안에 들어온 값으로 정상 데이터 입니다.
# -1 이면 outlier로 간주 합니다.

print(results)
#[ 1  1 -1  1  1  1  1  1  1  1 -1]
#[ 1  1 -1  1  1  1  1 -1 -1 -1 -1] 0.5
