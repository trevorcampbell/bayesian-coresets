import ehkmeans
import datasets

X = datasets.get_mnist()
print X.shape
print X[:, :10]


X = datasets.get_birch()

print X.shape
print X[:, :10]

X = datasets.get_covtype()

print X.shape
print X[:, :10]

X = datasets.get_sun()

print X.shape
print X[:, :10]




