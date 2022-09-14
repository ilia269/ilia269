from tensor import Tensor
import numpy as np
from modeles import Linear, ReLU, LeakyReLU, Sequential, Sigmoid
import torch
from optomizer import GD
from sklearn.datasets import make_blobs, make_moons
import random
from matplotlib import pyplot

# X, y = make_moons(400, noise=0.075)
# X_test, y_test = make_moons(400, noise=0.075)

X, y = make_blobs(400, 2)
X_test, y_test = make_blobs(400, 2)


# l1 = Linear(2, 3)
# l2 = Linear(3, 3)
# l3 = Sigmoid()
# l4 = Linear(3, 3)
# l5 = Linear(3, 1)
l6 = Sigmoid()

l1 = Linear(2, 1)

sequential = Sequential(l1, l6)
opt = GD(sequential.parameters(), 0.01)

dataset = list(zip(X, y))


def get_loss():
    res = 0
    for x, y in zip(X_test, y_test):
        p = sequential.predict(Tensor(x))
        if p == y:
            res += 1

    return res / 400


for i in range(20):
    random.shuffle(dataset)

    for x, y in dataset:
        answer = Tensor(y) - sequential(Tensor(x))
        res = answer @ answer.transpose()
        res.backward()
        opt.step()
        opt.zero_grads()

    print(f"iter is {i}   loss is {1-get_loss()}")


red = [x for x in X if sequential.predict(Tensor([x])) == 0]
blue = [x for x in X if sequential.predict(Tensor([x])) != 0]

red1 = [x[0] for x in red]
red2 = [x[1] for x in red]

blue1 = [x[0] for x in blue]
blue2 = [x[1] for x in blue]

pyplot.plot(red1, red2, 'bo')
pyplot.plot(blue1, blue2, 'ro')
pyplot.show()

