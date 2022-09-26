from tensor import Tensor, Parameter
import numpy as np
from modeles import Linear, ReLU, Sequential, LeakyReLU
import torch
from optomizer import GD, Adam
from sklearn.datasets import make_blobs, make_moons
import random
from matplotlib import pyplot
from memory_profiler import profile


def get_loss_for_classification(model, test_dataset):
    res = 0
    for x, y in test_dataset:
        p = model.predict(Tensor(x))
        if p == y:
            res += 1

    return res / len(test_dataset)


def get_mse_loss(model, dataset):
    x, y = get_x_y_from_dataset(dataset)
    model_results = np.array([float(model.forward(Tensor(x_i)).data) for x_i in x])
    true_results = np.array(y)

    tmp = (model_results - true_results)**2
    mse = tmp.sum()/len(dataset)

    return mse


def get_dataset_for_function(f, x_left=-10.0, x_right=10.0, n=400):
    step = (x_right - x_left) / n
    X = [x_left + i*step for i in range(n)]
    y = [f(x_i) for x_i in X]

    dataset = list(zip(X, y))
    random.shuffle(dataset)

    return dataset


def get_train_validate_test_dataset(full_dataset):
    n_train = int(0.6*len(full_dataset))
    n_validate = int(0.8*len(full_dataset))

    ds_train = full_dataset[:n_train]
    ds_validate = full_dataset[n_train:n_validate]
    ds_test = full_dataset[n_validate:]

    return ds_train, ds_validate, ds_test


def get_x_y_from_dataset(dataset):
    x = [item[0] for item in dataset]
    y = [item[1] for item in dataset]

    return x, y


def get_loss_str(iter, train_loss, validate_loss):
    iter = str(iter)
    train_loss = str(round(train_loss, 10))
    validate_loss = str(round(validate_loss, 10))

    gap1 = 5 - len(iter)
    gap2 = 15 - len(train_loss)

    res_str = iter + gap1*" " + train_loss + gap2*" " + validate_loss

    return res_str


def train(model, dataset, n_epochs, opt):
    train_ds, validate_ds, _ = get_train_validate_test_dataset(dataset)
    train_loss_values = [0]*n_epochs
    validation_loss_values = [0]*n_epochs

    for i in range(n_epochs):
        random.shuffle(train_ds)

        for x, y in train_ds:
            x = Tensor(x)
            loss = Tensor(y) - model(x)
            res = loss @ loss.transpose()
            res.backward()
            opt.step()
            opt.zero_grads()

        train_loss_values[i] = get_mse_loss(model, train_ds)
        validation_loss_values[i] = get_mse_loss(model, validate_ds)

        print(get_loss_str(i, get_mse_loss(model, train_ds), get_mse_loss(model, validate_ds)))

        # loss_str = get_loss_str(iter=i,
        #                         train_loss=get_mse_loss(model, train_ds),
        #                         validate_loss=get_mse_loss(model, validate_ds),
        #                         test_loss=get_mse_loss(model, test_ds))
        # print(loss_str)

    return train_loss_values, validation_loss_values


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


def show_loss(loss_values, validate_values):
    x = list(range(len(loss_values)))

    y1 = loss_values
    y2 = validate_values

    pyplot.plot(x, y1)
    pyplot.plot(x, y2)

    pyplot.show()


# Просто запуск своего обучения

l1 = Linear(16, 20)
l2 = ReLU()
l3 = Linear(20, 50)
l4 = ReLU()
l5 = Linear(50, 100)
l6 = ReLU()
l7 = Linear(100, 50)
l8 = ReLU()
l9 = Linear(50, 1)


model = Sequential(l1, l2, l3, l4, l5, l6, l7, l8, l9)

opt = Adam(model.parameters(), lr=0.0001)
target_f = lambda x: 3*x**2 - x**3 + 6
# dataset = get_dataset_for_function(target_f)

import json

with open("dataset.json", "r") as f:
    r = json.load(f)
    x = [r["x"][i] for i in range(0, 50000, 100)]
    y = [r["y"][i] for i in range(0, 50000, 100)]
    dataset = list(zip(x, y))


loss_values, validate_values = train(model, dataset, 50, opt)
show_loss(loss_values, validate_values)
_, _, test_ds = get_train_validate_test_dataset(dataset)
print(f"\nTest MSE is {get_mse_loss(model, test_ds)}")
print("Training is over")
