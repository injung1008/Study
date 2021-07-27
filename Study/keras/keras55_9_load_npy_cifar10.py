from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes
import numpy as np

x_train = np.load('./_save/_npy/k55_x_train_cifar10.npy')
y_train = np.load('./_save/_npy/k55_y_train_cifar10.npy')

x_test = np.load('./_save/_npy/k55_x_test_cifar10.npy')
y_test = np.load('./_save/_npy/k55_y_test_cifar10.npy')

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)






