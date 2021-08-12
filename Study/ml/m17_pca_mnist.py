import numpy as np     
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data() 

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

#실습 
#pca를 통해 0.95이상인 n_components 가 몇개인지? 

x = x.reshape(70000, 784)


pca = PCA(n_components=784) #총 10개 칼럼을 7개로 압축하겠다 
x = pca.fit_transform(x)

#주성분 분석이 어느정도 영향을 미치는지 알고 있다면 판단 하는게 쉬워짐 차원축소의 비율에 대해서 확인함 
pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
# print(cumsum) 

print(np.argmax(cumsum >= 0.94)+1) #134
import matplotlib.pyplot as plt     
plt.plot(cumsum)
plt.grid() 
plt.show()