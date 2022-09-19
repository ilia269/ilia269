from tensor import Tensor, Parameter
import numpy as np
from modeles import Linear, Sin, ReLU, Sequential, Sigmoid, Power
import torch
from optomizer import GD, Adam
from sklearn.datasets import make_blobs, make_moons
import random
from matplotlib import pyplot
import gc


def get_loss_for_classification(model, test_dataset):
    res = 0
    for x, y in test_dataset:
        p = model.predict(Tensor(x))
        if p == y:
            res += 1

    return res / len(test_dataset)


def get_mse_loss_regression(model, test_dataset):
    x_test, y = get_x_y_from_dataset(test_dataset)
    model_results = np.array([float(model.forward(Tensor(x)).data) for x in x_test])
    true_results = np.array(y)

    tmp = (model_results - true_results)**2
    mse = tmp.sum()/len(test_dataset)

    return mse


def get_dataset_for_function(f, x_left=-10.0, x_right=10.0, n=400):
    step = (x_right - x_left) / n
    X = [x_left + i*step for i in range(n)]
    y = [f(x_i) for x_i in X]

    dataset = list(zip(X, y))
    random.shuffle(dataset)

    return dataset


def get_train_test_dataset(full_dataset):
    n = int(0.8*len(full_dataset))
    ds_train = full_dataset[:n]
    ds_test = full_dataset[n:]

    return ds_train, ds_test


def get_x_y_from_dataset(dataset):
    x = [item[0] for item in dataset]
    y = [item[1] for item in dataset]

    return x, y


def train(model, ds_train, n_epochs, opt, ds_test):
    for i in range(n_epochs):
        random.shuffle(ds_train)

        for x, y in ds_train:
            x = Tensor(x)
            loss = Tensor(y) - model(x)
            res = loss @ loss.transpose()
            res.backward()
            opt.step()
            opt.zero_grads()
        print(f"iter is {i}   loss is {get_mse_loss_regression(model, ds_test)}")


def show_regression_results(model, target_function, ds_test, x_left=-10.0, x_right=10.0, n=400):
    step = (x_right-x_left) / n
    x_target_function = [x_left + i*step for i in range(n)]
    y_target_function = [target_function(x) for x in x_target_function]

    x_test, _ = get_x_y_from_dataset(ds_test)

    x_model_prediction = x_test

    y_model_prediction = [float(model.forward(Tensor([x])).data) for x in x_model_prediction]

    pyplot.plot(x_target_function, y_target_function)
    pyplot.plot(x_model_prediction, y_model_prediction, "bo")

    pyplot.show()


# Просто запуск своего обучения

l1 = Linear(1, 10)
l2 = ReLU()
l3 = Linear(10, 100)
l4 = ReLU()
l5 = Linear(100, 10)
l6 = ReLU()
l7 = Linear(10, 1)


model = Sequential(l1, l2, l3, l4, l5, l6, l7)

opt = Adam(model.parameters(), lr=0.001)
target_f = lambda x: 3*x**2 - x**3 + 6
dataset = get_dataset_for_function(target_f)
train_ds, test_ds = get_train_test_dataset(dataset)

train(model, train_ds, 100, opt, test_ds)

show_regression_results(model, target_f, test_ds)


# import json
#
# with open("dataset.json", "r") as f:
#     r = json.load(f)
#     x = [r["x"][i] for i in range(0, 50000, 100)]
#     y = [r["y"][i] for i in range(0, 50000, 100)]
#     print(len(x))
#     print(len(y))
#     pyplot.plot(x, y)
#     pyplot.show()


def show_regression_results_torch(model, target_function, ds_test, x_left=-10.0, x_right=10.0, n=400):
    step = (x_right-x_left) / n
    x_target_function = [x_left + i*step for i in range(n)]
    y_target_function = [target_function(x) for x in x_target_function]

    x_test, _ = get_x_y_from_dataset(ds_test)

    x_model_prediction = x_test
    y_model_prediction = [float(model.forward(torch.Tensor([x]))) for x in x_model_prediction]

    pyplot.plot(x_target_function, y_target_function)
    pyplot.plot(x_model_prediction, y_model_prediction, "bo")

    pyplot.show()

# Моё обновление весов, но тензоры из torch

# l1 = torch.nn.Linear(1, 3)
# l2 = torch.nn.ReLU()
# l3 = torch.nn.Linear(3, 3)
# l4 = torch.nn.ReLU()
# l5 = torch.nn.Linear(3, 3)
# l6 = torch.nn.ReLU()
# l7 = torch.nn.Linear(3, 1)
#
# model = torch.nn.Sequential(l1, l2, l3, l4, l5, l6, l7)
# target_f = lambda x: 3*x**2 - x**3 + 6
# dataset = get_dataset_for_function(target_f)
# train_ds, test_ds = get_train_test_dataset(dataset)
#
#
# for i in range(100):
#     random.shuffle(train_ds)
#     print(i)
#     # if i % 10 == 0:
#         # print(l1.weight)
#         # print(l3.weight)
#         # print(l5.weight)
#         # print(l7.weight)
#
#     for x, y in train_ds:
#         x = torch.Tensor([x])
#         loss = torch.Tensor([y]) - model(x)
#         res = loss * loss
#
#         res.backward()
#         w1 = l1.weight - 0.0001 * l1.weight.grad
#         w3 = l3.weight - 0.0001 * l3.weight.grad
#         w5 = l5.weight - 0.0001 * l5.weight.grad
#         w7 = l7.weight - 0.0001 * l7.weight.grad
#         b1 = l1.bias - 0.0001 * l1.bias.grad
#         b3 = l3.bias - 0.0001 * l3.bias.grad
#         b5 = l5.bias - 0.0001 * l5.bias.grad
#         b7 = l7.bias - 0.0001 * l7.bias.grad
#
#         l1.weight.data = w1.data
#         l3.weight.data = w3.data
#         l5.weight.data = w5.data
#         l7.weight.data = w7.data
#         l1.bias.data = b1.data
#         l3.bias.data = b3.data
#         l5.bias.data = b5.data
#         l7.bias.data = b7.data
#
#         l1.weight.grad *= 0.0
#         l3.weight.grad *= 0.0
#         l5.weight.grad *= 0.0
#         l7.weight.grad *= 0.0
#         l1.bias.grad *= 0.0
#         l3.bias.grad *= 0.0
#         l5.bias.grad *= 0.0
#         l7.bias.grad *= 0.0
#
#
# show_regression_results_torch(model, target_f, test_ds)

# Для проверки результатов
#
# torch_l1 = torch.nn.Linear(1, 3)
# torch_l2 = torch.nn.ReLU()
# torch_l3 = torch.nn.Linear(3, 3)
# torch_l4 = torch.nn.ReLU()
# torch_l5 = torch.nn.Linear(3, 1)
#
# my_l1 = Linear(1, 3)
# my_l2 = ReLU()
# my_l3 = Linear(3, 3)
# my_l4 = ReLU()
# my_l5 = Linear(3, 1)
#
#
# def turn_my_to_ones(x, alpha=1.0):
#     for x_i in x:
#         x_i.weight.data = alpha * x_i.weight.data / x_i.weight.data
#         x_i.bias.data = alpha * x_i.bias.data / x_i.bias.data
#
#
# def turn_torch_to_ones(x, alpha=1.0):
#     for x_i in x:
#         x_i.weight = torch.nn.Parameter(alpha * x_i.weight / x_i.weight)
#         x_i.bias = torch.nn.Parameter(alpha * x_i.bias / x_i.bias)
#
#
# torch_list = [torch_l1, torch_l3, torch_l5]
# turn_torch_to_ones(torch_list, 2.53)
#
# model1 = torch.nn.Sequential(torch_l1, torch_l2, torch_l3, torch_l4, torch_l5)
# my_list = [my_l1, my_l3, my_l5]
# turn_my_to_ones(my_list, 2.53)
#
# model2 = Sequential(my_l1, my_l2, my_l3, my_l4, my_l5)
#
# def take_step():
#     torch_res = model1(torch.Tensor([10.0]))
#     torch_res.backward()
#
#     my_res = model2(Tensor([10.0]))
#     my_res.backward()
#
#     return my_res, torch_res
#
#
# take_step()
# take_step()
# my_res, torch_res = take_step()

# print("Results:")
# print(my_res)
# print(torch_res)
# print("L 1:")
# print(torch_l1.weight.grad)
# print(my_l1.weight.grad)
# print("L 3:")
# print(torch_l3.weight.grad)
# print(my_l3.weight.grad)
# print("L 5:")
# print(torch_l5.weight.grad)
# print(my_l5.weight.grad)


# l1 = torch.nn.Linear(1, 10)
# l2 = torch.nn.ReLU()
# l3 = torch.nn.Linear(10, 10)
# l4 = torch.nn.ReLU()
# l5 = torch.nn.Linear(10, 10)
# l6 = torch.nn.ReLU()
# l7 = torch.nn.Linear(10, 1)
#
# model = torch.nn.Sequential(l1, l2, l3, l4, l5, l6, l7)
# loss = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# target_f = lambda x: 3*x**2 - x**3 + 6
# dataset = get_dataset_for_function(target_f)
# x, y = get_x_y_from_dataset(dataset)
#
# for i in range(100):
#     loss = None
#     for x_i, y_i in zip(x, y):
#         y_predicted = model(torch.Tensor([x_i]))
#         L = (y_predicted - torch.Tensor([y_i]))**2
#
#         optimizer.zero_grad()  # обнуляем градиенты
#         L.backward()  # вычисляем градиенты
#         optimizer.step()  # подправляем параметры
#
#         loss = L.data
#     print(f"i is {i} loss is {loss}")
#
# show_regression_results_torch(model, target_f, dataset)
