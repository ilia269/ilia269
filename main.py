from tensor import Tensor
import numpy as np
from modeles import Linear, ReLU, LeakyReLU, Sequential, Sigmoid
import torch

# _______________________
# a = Tensor(data=[[1, 1], [3, 4]]).transpose()
# b = Tensor(data=[1, 6])
# c = Tensor(data=[[3], [4]])
# x = Tensor(data=[1, 4])
# t = Tensor(data=[5, 7])
#
# r = np.matrix([[1], [2]])
# e = np.matrix([[3, 6]])
#
# def step():
#     l1.weight.data += l1.weight.grad.data*(-0.1)
#     l2.weight.data += l2.weight.grad.data*(-0.1)
#     l3.weight.data += l3.weight.grad.data*(-0.1)
#
#     l1.weight.zero_grad()
#     l2.weight.zero_grad()
#     l3.weight.zero_grad()
#
#
# l1 = Linear(2, 3)
# l2 = Linear(3, 4)
# l3 = Linear(4, 1)
# l4 = Sigmoid()
# model = Sequential(l1, l2, l3)
# res = model(x)
# res.backward()
# step()
# res = model(x)
# res.backward()
#
# for _ in range(50):
#     res = model(x)
#     print(res)
#     res.backward()
#     step()
# _______________________________________________

# a = Tensor(data=[[1, 1], [3, 4]]).transpose()
# b = Tensor(data=[1, 6])
# c = Tensor(data=[[3], [4]])
# x = Tensor(data=[1, 4])
# z = (x @ a + b) @ c
#
# for _ in range(10):
#     z.backward()
#     a = a + Tensor(data=[[-0.1, -0.1], [-0.1, -0.1]]) * a.grad
#     b = b + Tensor(data=[-0.1, -0.1]) * b.grad
#     c = c + Tensor(data=[[-0.1], [-0.1]]) * c.grad
#     z = (x @ a**2 + b) @ c
#     a.grad, b.grad, c.grad = Tensor([[0, 0], [0, 0]]), Tensor([0, 0]), Tensor([[0], [0]])
#     print(z)

x = Tensor([10])
b = Tensor([5])
t = x**2 + b


for _ in range(50):
    print(f"t is {t}")
    print(f"x is {x}\n")
    t.backward()
    x.data = (-0.01)*x.grad.data
    x.grad.data *= 0.0
    t = x**2 + b
    t.backward()

