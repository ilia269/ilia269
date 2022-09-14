from tensor import Tensor
import numpy as np
from modeles import Linear, ReLU, LeakyReLU, Sequential, Sigmoid
import torch


# _______________________
a = Tensor(data=[[1, 1], [3, 4]]).transpose()
b = Tensor(data=[1, 6])
c = Tensor(data=[[3], [4]])
x = Tensor(data=[1, 4])
t = Tensor(data=[5, 7])

r = np.matrix([[1], [2]])
e = np.matrix([[3, 6]])


# def step():
#     l1.weight.data += l1.weight.grad.data*(-0.1)
#     l2.weight.data += l2.weight.grad.data*(-0.1)
#     l4.weight.data += l4.weight.grad.data*(-0.1)
#
#     l1.weight.zero_grad()
#     l2.weight.zero_grad()
#     l4.weight.zero_grad()
#
#
# l1 = Linear(2, 3)
# l2 = Linear(3, 4)
# l3 = Sigmoid()
# l4 = Linear(4, 1)
# l5 = Sigmoid()
# model = Sequential(l1, l2, l3, l4, l5)
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


l1 = torch.nn.Linear(2, 3)
l2 = torch.nn.Linear(3, 4)
l3 = torch.nn.Sigmoid()
l4 = torch.nn.Linear(4, 1)
l5 = torch.nn.Sigmoid()
model = torch.nn.Sequential(l1, l2, l3, l4, l5)

x = torch.Tensor([1, 4])

res = model(x)
optimizer = torch.optim.SGD([x], lr=1, momentum=0.5)
for it in range(20):
    optimizer.zero_grad()  # обнуляем градиенты
    res = model(torch.Tensor([1, 4]))  # вычисляем значения функции
    res.backward()  # вычисляем градиенты
    optimizer.step()  # подправляем параметры

    print(res)

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
# _______________________________________________________
# x = Tensor([10])
# b = Tensor([5])
# t = x**2 + b
#
#
# for _ in range(50):
#     print(f"t is {t}")
#     print(f"x is {x}\n")
#     t.backward()
#     x.data = (-0.01)*x.grad.data
#     x.grad.data *= 0.0
#     t = x**2 + b
#     t.backward()