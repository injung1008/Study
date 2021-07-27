# csv와 numpy 로드 속도 차이 많이 난다.

from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100


iris = load_iris()
boston = load_boston()
cancer = load_breast_cancer()
diabetes = load_diabetes()
wine = load_wine()

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = fashion_mnist.load_data()
(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

# iris
iris_x_data = iris.data
iris_y_data = iris.target


print(type(iris_x_data), type(iris_y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(iris_x_data.shape, iris_y_data.shape)
# (150, 4) (150,)

np.save('./_save/_npy/k55_x_data_iris.npy', arr=iris_x_data)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=iris_y_data)

# boston
boston_x_data = boston.data
boston_y_data = boston.target

np.save('./_save/_npy/k55_x_data_boston.npy', arr=boston_x_data)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=boston_y_data)


# cancer
cancer_x_data = cancer.data
cancer_y_data = cancer.target

np.save('./_save/_npy/k55_x_data_cancer.npy', arr=cancer_x_data)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=cancer_y_data)

# diabetes
diabetes_x_data = diabetes.data
diabetes_y_data = diabetes.target

np.save('./_save/_npy/k55_x_data_diabetes.npy', arr=diabetes_x_data)
np.save('./_save/_npy/k55_y_data_diabetes.npy', arr=diabetes_y_data)

# wine
wine_x_data = wine.data
wine_y_data = wine.target

np.save('./_save/_npy/k55_x_data_wine.npy', arr=wine_x_data)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=wine_y_data)

# mnist
np.save('./_save/_npy/k55_x_train_mnist.npy', arr=mnist_x_train)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=mnist_y_train)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=mnist_x_test)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=mnist_y_test)

# fashion
np.save('./_save/_npy/k55_x_train_fashion.npy', arr=fashion_x_train)
np.save('./_save/_npy/k55_y_train_fashion.npy', arr=fashion_y_train)
np.save('./_save/_npy/k55_x_test_fashion.npy', arr=fashion_x_test)
np.save('./_save/_npy/k55_y_test_fashion.npy', arr=fashion_y_test)

# cifar10
np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=cifar10_x_train)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=cifar10_y_train)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=cifar10_x_test)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=cifar10_y_test)


# cifar100
np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=cifar100_x_train)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=cifar100_y_train)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=cifar100_x_test)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=cifar100_y_test)
